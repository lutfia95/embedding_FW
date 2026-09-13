from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from typing import Any

from .errors import ValidationError
from .metrics import cosine_distance, cosine_similarity, pairwise
from .models import DiagnosticSpec, RepresentationMethod

_VECTOR_METHODS = {
    RepresentationMethod.MEAN,
    RepresentationMethod.MAX,
    RepresentationMethod.NATIVE,
}


def diagnose_embeddings(
    outputs: dict[str, dict[str, Any]],
    sequence_order: list[str],
    spec: DiagnosticSpec,
) -> dict[str, Any]:
    """Diagnose pooled embedding geometry without changing source values."""
    if spec.representation not in _VECTOR_METHODS:
        raise ValidationError("Diagnostics require a mean, max, or native vector representation")
    if spec.sample_pairs < 1:
        raise ValidationError("sample_pairs must be positive")
    if spec.hubness_k < 1:
        raise ValidationError("hubness_k must be positive")
    if spec.max_exact_hubness_sequences < 2:
        raise ValidationError("max_exact_hubness_sequences must be at least two")

    results = {}
    shared_pairs = _sample_pair_indices(len(sequence_order), spec.sample_pairs, spec.seed)
    for model_id, output in outputs.items():
        selected, selection_warnings = _select_vectors(output, sequence_order, spec)
        if selected is None:
            results[model_id] = {
                "available": False,
                "reason": (
                    f"Complete {spec.representation} representations were not found"
                    + (f" at layer {spec.layer}" if spec.layer is not None else "")
                ),
                "warnings": selection_warnings,
            }
            continue
        layer, vectors = selected
        results[model_id] = _diagnose_one(
            vectors,
            sequence_order,
            shared_pairs,
            layer=layer,
            spec=spec,
            warnings=selection_warnings,
        )

    return {
        "available": any(item.get("available", False) for item in results.values()),
        "representation": str(spec.representation),
        "requested_layer": spec.layer,
        "sequence_order": sequence_order,
        "model_order": list(outputs),
        "sample_pair_seed": spec.seed,
        "sample_pair_count": len(shared_pairs),
        "models": results,
        "definitions": {
            "anisotropy_score": (
                "mean sampled pairwise cosine divided by 1/sqrt(embedding_dimension); "
                "a descriptive geometry diagnostic, not a universal biological threshold"
            ),
            "participation_ratio": (
                "square(sum(per-dimension variance)) divided by sum(square(per-dimension variance))"
            ),
            "distance_concentration": "distribution of sampled pairwise cosine distances",
            "robin_hood_index": (
                "fraction of total k-occurrences that must move from above-mean nodes "
                "to make neighbor occurrence uniform"
            ),
        },
        "scientific_guardrail": (
            "Diagnostics describe embedding geometry and do not establish biological quality"
        ),
    }


def _select_vectors(
    output: dict[str, Any], sequence_order: list[str], spec: DiagnosticSpec
) -> tuple[tuple[str | int, list[list[float]]] | None, list[str]]:
    groups: dict[tuple[str, str | int], dict[str, list[float]]] = {}
    for item in output.get("representations", []):
        if item.get("method") != str(spec.representation):
            continue
        layer = item.get("layer", "model_default")
        if spec.layer is not None and layer != spec.layer:
            continue
        groups.setdefault((item["method"], layer), {})[item.get("sequence_id")] = item.get("values")
    complete = [
        (key, mapping)
        for key, mapping in groups.items()
        if all(sequence_id in mapping for sequence_id in sequence_order)
    ]
    if not complete:
        return None, []
    warnings = []
    if spec.layer is None and len(complete) > 1:
        warnings.append(
            f"Multiple complete layers were present; selected the first layer {complete[0][0][1]}"
        )
    (method, layer), mapping = complete[0]
    del method
    vectors = [_clean_vector(mapping[sequence_id], sequence_id) for sequence_id in sequence_order]
    dimension = len(vectors[0]) if vectors else 0
    if not vectors or dimension == 0:
        raise ValidationError("Diagnostics require at least one non-empty vector")
    if any(len(vector) != dimension for vector in vectors):
        raise ValidationError(
            "Diagnostic vectors must share one embedding dimension",
            context={"layer": layer},
        )
    return (layer, vectors), warnings


def _clean_vector(values: Any, sequence_id: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValidationError(
            "Diagnostic representation must be a non-empty vector",
            context={"sequence_id": sequence_id},
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                "Diagnostic representation contains a non-numeric value",
                context={"sequence_id": sequence_id},
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError(
                "Diagnostic representation contains a non-finite value",
                context={"sequence_id": sequence_id},
            )
        result.append(number)
    return result


def _diagnose_one(
    vectors: list[list[float]],
    sequence_order: list[str],
    sampled_pairs: list[tuple[int, int]],
    *,
    layer: str | int,
    spec: DiagnosticSpec,
    warnings: list[str],
) -> dict[str, Any]:
    n, dimension = len(vectors), len(vectors[0])
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    means = [sum(vector[column] for vector in vectors) / n for column in range(dimension)]
    variances = [
        sum((vector[column] - means[column]) ** 2 for vector in vectors) / n
        for column in range(dimension)
    ]
    variance_sum = sum(variances)
    variance_square_sum = sum(value * value for value in variances)
    participation_ratio = (
        variance_sum * variance_sum / variance_square_sum if variance_square_sum else None
    )

    raw_cosines = [cosine_similarity(vectors[i], vectors[j]) for i, j in sampled_pairs]
    centered = [
        [value - means[column] for column, value in enumerate(vector)] for vector in vectors
    ]
    centered_cosines = [cosine_similarity(centered[i], centered[j]) for i, j in sampled_pairs]
    distances = [cosine_distance(vectors[i], vectors[j]) for i, j in sampled_pairs]
    mean_cosine = _mean(raw_cosines)
    expected_cosine_std = 1.0 / math.sqrt(dimension)

    duplicate_groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    for sequence_id, vector in zip(sequence_order, vectors, strict=True):
        duplicate_groups[tuple(vector)].append(sequence_id)
    duplicates = [ids for ids in duplicate_groups.values() if len(ids) > 1]

    model_warnings = list(warnings)
    zero_count = sum(norm == 0 for norm in norms)
    if zero_count:
        model_warnings.append(
            f"{zero_count} zero vectors were found; their cosine similarities are reported as zero"
        )
    if duplicates:
        model_warnings.append(
            f"{sum(len(group) - 1 for group in duplicates)} duplicate vectors found"
        )
    if n < 3:
        model_warnings.append("Geometry diagnostics based on fewer than three vectors are unstable")

    if n > spec.max_exact_hubness_sequences:
        hubness = {
            "available": False,
            "reason": (
                f"Exact hubness is limited to {spec.max_exact_hubness_sequences} sequences; "
                "an approximate backend is not yet implemented"
            ),
        }
    elif n < 2:
        hubness = {"available": False, "reason": "Hubness requires at least two vectors"}
    else:
        requested_k = spec.hubness_k
        k = min(requested_k, n - 1)
        if k != requested_k:
            model_warnings.append(f"hubness_k was reduced from {requested_k} to {k}")
        similarity = pairwise(vectors, "cosine_similarity")
        occurrences = [0] * n
        for row_index, row in enumerate(similarity):
            neighbors = sorted(
                (index for index in range(n) if index != row_index),
                key=lambda index: (-row[index], index),
            )[:k]
            for neighbor in neighbors:
                occurrences[neighbor] += 1
        total_occurrences = sum(occurrences)
        mean_occurrence = total_occurrences / n
        robin_hood = (
            sum(value - mean_occurrence for value in occurrences if value > mean_occurrence)
            / total_occurrences
            if total_occurrences
            else 0.0
        )
        top_hubs = sorted(
            (
                {"sequence_id": sequence_order[index], "k_occurrence": count}
                for index, count in enumerate(occurrences)
            ),
            key=lambda item: (-item["k_occurrence"], item["sequence_id"]),
        )[: min(20, n)]
        hubness = {
            "available": True,
            "k": k,
            "robin_hood_index": robin_hood,
            "occurrence_min": min(occurrences),
            "occurrence_mean": mean_occurrence,
            "occurrence_max": max(occurrences),
            "top_hubs": top_hubs,
        }

    return {
        "available": True,
        "layer": layer,
        "sequence_count": n,
        "embedding_dimension": dimension,
        "norms": {
            "minimum": min(norms),
            "mean": _mean(norms),
            "maximum": max(norms),
            "zero_count": zero_count,
        },
        "dimensions": {
            "means": means,
            "variances": variances,
            "mean_variance": _mean(variances),
            "variance_std": _population_std(variances),
            "participation_ratio": participation_ratio,
        },
        "duplicates": {
            "duplicate_vector_count": sum(len(group) - 1 for group in duplicates),
            "groups": duplicates[:20],
            "groups_truncated": len(duplicates) > 20,
        },
        "anisotropy": {
            "sample_pair_count": len(sampled_pairs),
            "average_pairwise_cosine": mean_cosine,
            "expected_isotropic_cosine_std": expected_cosine_std,
            "anisotropy_score": (
                mean_cosine / expected_cosine_std if mean_cosine is not None else None
            ),
            "mean_centered_average_pairwise_cosine": _mean(centered_cosines),
        },
        "distance_concentration": {
            "metric": "cosine_distance",
            "sample_pair_count": len(distances),
            **_distribution(distances),
        },
        "hubness": hubness,
        "warnings": model_warnings,
    }


def _sample_pair_indices(n: int, limit: int, seed: int) -> list[tuple[int, int]]:
    pair_count = n * (n - 1) // 2
    if pair_count == 0:
        return []
    if pair_count <= limit:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    ranks = sorted(random.Random(seed).sample(range(pair_count), limit))
    prefixes = [i * (2 * n - i - 1) // 2 for i in range(n)]
    result = []
    for rank in ranks:
        i = bisect.bisect_right(prefixes, rank) - 1
        j = i + 1 + rank - prefixes[i]
        result.append((i, j))
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _population_std(values: list[float]) -> float | None:
    mean = _mean(values)
    if mean is None:
        return None
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _distribution(values: list[float]) -> dict[str, float | None]:
    mean = _mean(values)
    std = _population_std(values)
    return {
        "minimum": min(values) if values else None,
        "mean": mean,
        "maximum": max(values) if values else None,
        "standard_deviation": std,
        "coefficient_of_variation": std / mean if mean not in (None, 0.0) else None,
    }
