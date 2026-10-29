from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

from .diagnostics import diagnose_embeddings
from .errors import ValidationError
from .models import (
    DiagnosticSpec,
    ImportedNormalization,
    RepresentationMethod,
    TransformationMethod,
    TransformationSpec,
)

_VECTOR_METHODS = {
    RepresentationMethod.MEAN,
    RepresentationMethod.MAX,
    RepresentationMethod.NATIVE,
}


def transform_embeddings(
    outputs: dict[str, dict[str, Any]],
    sequence_order: list[str],
    spec: TransformationSpec,
) -> dict[str, Any]:
    """Create derived embedding spaces without modifying the source outputs."""
    if spec.method != TransformationMethod.MEAN_CENTER:
        raise ValidationError(f"Unsupported transformation: {spec.method}")
    if spec.representation not in _VECTOR_METHODS:
        raise ValidationError("Mean-centering requires a mean, max, or native representation")
    if not sequence_order:
        raise ValidationError("Transformation requires at least one sequence")
    if len(sequence_order) != len(set(sequence_order)):
        raise ValidationError("Transformation sequence IDs must be unique")
    if spec.sample_pairs < 1 or spec.hubness_k < 1:
        raise ValidationError("Diagnostic sample_pairs and hubness_k must be positive")

    fit_ids = (
        list(spec.fit_sequence_ids) if spec.fit_sequence_ids is not None else sequence_order[:]
    )
    if not fit_ids:
        raise ValidationError("fit_sequence_ids cannot be empty")
    if len(fit_ids) != len(set(fit_ids)):
        raise ValidationError("fit_sequence_ids must be unique")
    unknown_fit_ids = sorted(set(fit_ids) - set(sequence_order))
    if unknown_fit_ids:
        raise ValidationError(
            "fit_sequence_ids contains IDs absent from the source job",
            context={"unknown_sequence_ids": unknown_fit_ids},
        )

    transformed_models: dict[str, dict[str, Any]] = {}
    source_models: dict[str, dict[str, Any]] = {}
    parameters: dict[str, dict[str, Any]] = {}
    failures = []
    source_model_map = {}

    for model_id, output in outputs.items():
        selected = _select_representation(output, sequence_order, spec)
        if selected is None:
            failures.append(
                {
                    "model": model_id,
                    "code": "representation_unavailable",
                    "message": (
                        f"Complete {spec.representation} representations were not found"
                        + (f" at layer {spec.layer}" if spec.layer is not None else "")
                    ),
                }
            )
            continue
        layer, source_items, warnings = selected
        vectors = [_clean_vector(item["values"], item["sequence_id"]) for item in source_items]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValidationError(
                "Vectors must have a consistent dimension before mean-centering",
                context={"model": model_id, "layer": layer},
            )
        index_by_id = {sequence_id: index for index, sequence_id in enumerate(sequence_order)}
        fit_vectors = [vectors[index_by_id[sequence_id]] for sequence_id in fit_ids]
        fitted_mean = [
            sum(vector[column] for vector in fit_vectors) / len(fit_vectors)
            for column in range(dimension)
        ]
        centered_vectors = [
            [value - fitted_mean[column] for column, value in enumerate(vector)]
            for vector in vectors
        ]
        derived_id = f"{model_id}::mean-centered"
        if derived_id in transformed_models:
            raise ValidationError(f"Derived model identifier collision: {derived_id}")

        parameter_payload = {
            "method": str(spec.method),
            "source_model": model_id,
            "derived_model": derived_id,
            "representation": str(spec.representation),
            "layer": layer,
            "fit_sequence_ids": fit_ids,
            "fit_scope": "explicit_subset"
            if spec.fit_sequence_ids is not None
            else "all_sequences",
            "mean_vector": fitted_mean,
        }
        parameter_bytes = json.dumps(
            parameter_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        parameter_payload["sha256"] = hashlib.sha256(parameter_bytes).hexdigest()
        parameters[derived_id] = parameter_payload

        transformed_items = []
        for source_item, centered in zip(source_items, centered_vectors, strict=True):
            source_metadata = source_item.get("metadata") or {}
            transformed_items.append(
                {
                    "sequence_id": source_item["sequence_id"],
                    "method": str(spec.representation),
                    "layer": layer,
                    "values": centered,
                    "shape": [dimension],
                    "norm_before": math.sqrt(sum(value * value for value in centered)),
                    "normalization": str(ImportedNormalization.MEAN_CENTERED),
                    "metadata": {
                        **source_metadata,
                        "source_normalization": source_item.get("normalization", "unknown"),
                        "transformation": str(spec.method),
                        "transformation_parameters_sha256": parameter_payload["sha256"],
                    },
                }
            )

        manifest = deepcopy(output.get("manifest", {}))
        source_display_name = manifest.get("display_name", model_id)
        manifest.update(
            {
                "id": derived_id,
                "display_name": f"{source_display_name} (mean-centered)",
                "representations": [str(spec.representation)],
                "default_layer": layer,
                "embedding_dimension": dimension,
                "conformance": f"derived_from:{manifest.get('conformance', 'unreported')}",
                "backend": "derived_transformation",
                "limitations": [
                    *manifest.get("limitations", []),
                    "Mean vector was fitted on the sequence IDs recorded in transformation provenance",
                ],
            }
        )
        model_provenance = {
            "operation": str(spec.method),
            "source_model": model_id,
            "derived_model": derived_id,
            "source_provenance": output.get("provenance", {}),
            "parameters_sha256": parameter_payload["sha256"],
            "fit_scope": parameter_payload["fit_scope"],
            "fit_sequence_count": len(fit_ids),
            "warnings": warnings,
        }
        transformed_models[derived_id] = {
            "manifest": manifest,
            "availability": {"available": True, "reason_code": None, "detail": None},
            "representations": transformed_items,
            "matrices": {},
            "residue_pairs": {},
            "provenance": model_provenance,
        }
        source_models[model_id] = output
        source_model_map[model_id] = derived_id

    if not transformed_models:
        raise ValidationError(
            "No source model contained the requested complete representation",
            context={"representation": str(spec.representation), "layer": spec.layer},
        )

    diagnostic_spec = DiagnosticSpec(
        representation=spec.representation,
        layer=spec.layer,
        sample_pairs=spec.sample_pairs,
        hubness_k=spec.hubness_k,
        seed=spec.seed,
    )
    before = diagnose_embeddings(source_models, sequence_order, diagnostic_spec)
    after = diagnose_embeddings(
        transformed_models,
        sequence_order,
        DiagnosticSpec(
            representation=spec.representation,
            layer=next(iter(parameters.values()))["layer"] if len(parameters) == 1 else None,
            sample_pairs=spec.sample_pairs,
            hubness_k=spec.hubness_k,
            seed=spec.seed,
        ),
    )
    warnings = []
    if spec.fit_sequence_ids is None:
        warnings.append(
            "The mean was fitted on all source sequences; use an explicit training/reference subset "
            "for held-out benchmark evaluation"
        )
    return {
        "schema_version": "1.0",
        "sequence_order": sequence_order,
        "models": transformed_models,
        "failures": failures,
        "transformation": {
            "method": str(spec.method),
            "representation": str(spec.representation),
            "requested_layer": spec.layer,
            "fit_sequence_ids": fit_ids,
            "fit_scope": "explicit_subset"
            if spec.fit_sequence_ids is not None
            else "all_sequences",
            "source_model_map": source_model_map,
            "warnings": warnings,
        },
        "transformation_parameters": parameters,
        "diagnostics": {"before": before, "after": after},
        "requested_visualizations": [],
        "provenance": {
            "operation": str(spec.method),
            "specification": {
                "representation": str(spec.representation),
                "layer": spec.layer,
                "sample_pairs": spec.sample_pairs,
                "hubness_k": spec.hubness_k,
                "seed": spec.seed,
            },
            "fit_scope": "explicit_subset"
            if spec.fit_sequence_ids is not None
            else "all_sequences",
            "fit_sequence_ids": fit_ids,
            "parameter_checksums": {
                model_id: value["sha256"] for model_id, value in parameters.items()
            },
        },
    }


def _select_representation(
    output: dict[str, Any], sequence_order: list[str], spec: TransformationSpec
) -> tuple[str | int, list[dict[str, Any]], list[str]] | None:
    groups: dict[tuple[str, str | int], dict[str, dict[str, Any]]] = {}
    for item in output.get("representations", []):
        if item.get("method") != str(spec.representation):
            continue
        layer = item.get("layer", "model_default")
        if spec.layer is not None and layer != spec.layer:
            continue
        groups.setdefault((item["method"], layer), {})[item.get("sequence_id")] = item
    complete = [
        (key, mapping)
        for key, mapping in groups.items()
        if all(sequence_id in mapping for sequence_id in sequence_order)
    ]
    if not complete:
        return None
    warnings = []
    if spec.layer is None and len(complete) > 1:
        warnings.append(
            f"Multiple complete layers were present; selected the first layer {complete[0][0][1]}"
        )
    (_, layer), mapping = complete[0]
    return layer, [mapping[sequence_id] for sequence_id in sequence_order], warnings


def _clean_vector(values: Any, sequence_id: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValidationError(
            "Mean-centering requires non-empty vectors",
            context={"sequence_id": sequence_id},
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                "Mean-centering encountered a non-numeric vector value",
                context={"sequence_id": sequence_id},
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError(
                "Mean-centering encountered a non-finite vector value",
                context={"sequence_id": sequence_id},
            )
        result.append(number)
    return result
