from __future__ import annotations

import platform
from dataclasses import asdict
from datetime import UTC, datetime

from . import __version__
from .cross_model import RESIDUE_METHODS, compare_cross_model
from .errors import FrameworkError, ResourceLimitError, ValidationError
from .metrics import (
    SEQUENCE_METRICS,
    VECTOR_METRICS,
    align_similarity,
    pairwise,
    pairwise_sequences,
    residue_similarity,
    sinkhorn_similarity,
    symmetric_best_match,
)
from .models import CompareRequest, EmbeddingRequest, RepresentationMethod, Visualization
from .registry import ModelRegistry, default_registry
from .representations import materialize
from .sequences import canonicalize_many, digest


class Pipeline:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or default_registry

    def estimate(self, request: EmbeddingRequest) -> dict:
        estimates = []
        for model_id in request.models:
            adapter = self.registry.adapter(model_id)
            sequences = canonicalize_many(request.sequences, adapter.manifest.alphabet)
            estimates.append(adapter.estimate_resources(sequences))
        n = len(request.sequences)
        return {
            "models": estimates,
            "sequence_count": n,
            "pair_count": n * (n - 1) // 2,
            "matrix_bytes_float64_per_metric": n * n * 8,
        }

    def embed(self, request: EmbeddingRequest) -> dict:
        if not request.models:
            raise ValidationError("At least one model is required")
        started = datetime.now(UTC)
        model_outputs = {}
        failures = []
        for requested_id in request.models:
            try:
                adapter = self.registry.adapter(requested_id)
                sequences = canonicalize_many(request.sequences, adapter.manifest.alphabet)
                adapter.validate_request(sequences, request.representations)
                per_layer = {}
                for layer in dict.fromkeys(spec.layer for spec in request.representations):
                    per_layer[layer] = adapter.embed(
                        sequences,
                        layer=layer,
                        device=request.execution.device,
                        precision=request.execution.precision,
                    )
                results = []
                for spec in request.representations:
                    raw_by_id = {raw.sequence_id: raw for raw in per_layer[spec.layer]}
                    for sequence in sequences:
                        results.append(asdict(materialize(raw_by_id[sequence.id], sequence, spec)))
                model_outputs[adapter.manifest.id] = {
                    "manifest": adapter.manifest.to_dict(),
                    "availability": asdict(adapter.availability()),
                    "representations": results,
                    "provenance": adapter.provenance(),
                }
            except FrameworkError as exc:
                if not request.execution.allow_partial:
                    raise
                failures.append({"model": requested_id, **exc.as_dict()})
            except Exception as exc:
                if not request.execution.allow_partial:
                    raise
                failures.append(
                    {"model": requested_id, "code": "unexpected_model_failure", "message": str(exc)}
                )
        finished = datetime.now(UTC)
        return {
            "schema_version": "1.0",
            "sequence_order": [sequence.id for sequence in request.sequences],
            "models": model_outputs,
            "failures": failures,
            "provenance": self._run_provenance(request, started, finished),
        }

    def compare(self, request: CompareRequest) -> dict:
        n = len(request.sequences)
        if n < 2 and not request.cross_model.enabled:
            raise ValidationError(
                "Comparison requires at least two sequences unless cross_model is enabled"
            )
        requested_methods = {spec.method for spec in request.representations}
        requested_metrics = {str(metric) for metric in request.metrics}
        cross_methods = set(request.cross_model.methods)
        vector_methods = {
            RepresentationMethod.MEAN,
            RepresentationMethod.MAX,
            RepresentationMethod.NATIVE,
        }
        if requested_metrics & set(VECTOR_METRICS) and not requested_methods & vector_methods:
            raise ValidationError(
                "Vector metrics require mean, max, or native sequence representations",
                context={"metrics": sorted(requested_metrics & set(VECTOR_METRICS))},
            )
        residue_metric_names = {
            "residue_global_alignment",
            "residue_local_alignment",
            "residue_best_match",
            "residue_optimal_transport",
        }
        if (
            requested_metrics & residue_metric_names
            and RepresentationMethod.RESIDUE not in requested_methods
        ):
            raise ValidationError(
                "Residue metrics require a residue representation",
                context={"metrics": sorted(requested_metrics & residue_metric_names)},
            )
        if request.cross_model.enabled:
            if len(request.models) < 2:
                raise ValidationError(
                    "Cross-model comparison requires at least two requested models"
                )
            resolved_models = {self.registry.resolve_id(model_id) for model_id in request.models}
            if len(resolved_models) < 2:
                raise ValidationError(
                    "Cross-model comparison requires at least two distinct resolved models"
                )
            if (
                cross_methods - RESIDUE_METHODS
                and request.cross_model.representation not in requested_methods
            ):
                raise ValidationError(
                    "The cross-model sequence representation must also be requested",
                    context={"representation": str(request.cross_model.representation)},
                )
            if (
                cross_methods & RESIDUE_METHODS
                and RepresentationMethod.RESIDUE not in requested_methods
            ):
                raise ValidationError(
                    "residue_cka and residue_rsa require the residue representation"
                )
        if request.execution.all_to_all == "exact" and n > request.execution.max_exact_sequences:
            raise ResourceLimitError(
                f"Exact comparison limit is {request.execution.max_exact_sequences} sequences",
                context={"sequence_count": n},
            )
        embedded = self.embed(request)
        sequences = [item.sequence.replace(" ", "").upper() for item in request.sequences]
        baselines = {}
        for metric in request.metrics:
            if str(metric) in SEQUENCE_METRICS:
                baselines[str(metric)] = pairwise_sequences(sequences, str(metric))
        for output in embedded["models"].values():
            matrices = {}
            grouped = {}
            for result in output["representations"]:
                key = f"{result['method']}@{result['layer']}"
                grouped.setdefault(key, []).append(result)
            for representation_key, results in grouped.items():
                method = results[0]["method"]
                if method in {
                    str(RepresentationMethod.MEAN),
                    str(RepresentationMethod.MAX),
                    str(RepresentationMethod.NATIVE),
                }:
                    for metric in request.metrics:
                        if str(metric) in VECTOR_METRICS:
                            matrices[f"{representation_key}:{metric}"] = pairwise(
                                [item["values"] for item in results], str(metric)
                            )
            residue_results = next(
                (results for key, results in grouped.items() if key.startswith("residue@")), None
            )
            residue_pairs = {}
            residue_metrics = requested_metrics & residue_metric_names
            if residue_metrics:
                if residue_results is None:
                    raise ValidationError(
                        "Residue comparison metrics require the residue representation"
                    )
                for i in range(n):
                    for j in range(i + 1, n):
                        similarity = residue_similarity(
                            residue_results[i]["values"], residue_results[j]["values"]
                        )
                        pair_result = {}
                        if "residue_global_alignment" in residue_metrics:
                            pair_result["residue_global_alignment"] = align_similarity(
                                similarity, local=False
                            )
                        if "residue_local_alignment" in residue_metrics:
                            pair_result["residue_local_alignment"] = align_similarity(
                                similarity, local=True
                            )
                        if "residue_best_match" in residue_metrics:
                            pair_result["residue_best_match"] = symmetric_best_match(similarity)
                        if "residue_optimal_transport" in residue_metrics:
                            pair_result["residue_optimal_transport"] = sinkhorn_similarity(
                                similarity
                            )
                        if Visualization.RESIDUE_SIMILARITY_MAP in request.visualizations:
                            pair_result["similarity_matrix"] = similarity
                        pair_result["similarity_shape"] = [len(similarity), len(similarity[0])]
                        residue_pairs[f"{request.sequences[i].id}::{request.sequences[j].id}"] = (
                            pair_result
                        )
            output["matrices"] = matrices
            output["residue_pairs"] = residue_pairs
        embedded["sequence_baselines"] = baselines
        embedded["metric_definitions"] = {
            "sequence_identity": "Needleman-Wunsch; match=1, mismatch=-1, linear gap=-1; identity per alignment column",
            "kmer_jaccard": "Jaccard similarity of unique 3-mers",
            "blosum62_global": "Needleman-Wunsch BLOSUM62; linear gap=-4; normalized by geometric mean of self scores",
            "blosum62_local": "Smith-Waterman BLOSUM62; linear gap=-4; normalized by geometric mean of self scores",
            "residue_optimal_transport": "Balanced entropy-regularized Sinkhorn transport with uniform residue mass",
        }
        embedded["cross_model"] = compare_cross_model(
            embedded["models"], embedded["sequence_order"], request.cross_model
        )
        spearman = embedded["cross_model"].get("comparisons", {}).get("matrix_spearman", {})
        embedded["model_agreement"] = (
            {
                "model_order": spearman.get("model_order", []),
                "spearman": spearman.get("values", []),
            }
            if spearman.get("available")
            else {"model_order": [], "spearman": []}
        )
        embedded["requested_visualizations"] = [str(item) for item in request.visualizations]
        return embedded

    @staticmethod
    def _run_provenance(request, started, finished) -> dict:
        provenance = {
            "framework_version": __version__,
            "schema_version": "1.0",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "sequence_digests": {
                item.id: {
                    "sha256": digest(item.sequence.replace(" ", "").upper()),
                    "length": len(item.sequence.replace(" ", "")),
                }
                for item in request.sequences
            },
            "execution": asdict(request.execution),
        }
        if isinstance(request, CompareRequest):
            provenance["cross_model"] = asdict(request.cross_model)
        return provenance
