from __future__ import annotations

import importlib.util
import math

from .errors import ValidationError
from .metrics import matrix_spearman, pairwise, pearson, upper_triangle
from .models import CrossModelMethod, CrossModelSpec, RepresentationMethod

SEQUENCE_RELATIONSHIP_METHODS = {
    CrossModelMethod.MATRIX_SPEARMAN,
    CrossModelMethod.MATRIX_PEARSON,
    CrossModelMethod.NEIGHBOR_OVERLAP,
    CrossModelMethod.CONSENSUS_RANK,
    CrossModelMethod.PAIR_DISAGREEMENT,
}
RESIDUE_METHODS = {CrossModelMethod.RESIDUE_CKA, CrossModelMethod.RESIDUE_RSA}


def linear_cka(left: list[list[float]], right: list[list[float]]) -> float | None:
    """Linear centered-kernel alignment; observations are rows, features are columns."""
    if len(left) != len(right) or len(left) < 2:
        return None
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        x = np.asarray(left, dtype=float)
        y = np.asarray(right, dtype=float)
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)
        cross = x.T @ y
        numerator = float(np.square(cross).sum())
        denominator = math.sqrt(float(np.square(x.T @ x).sum()) * float(np.square(y.T @ y).sum()))
        return numerator / denominator if denominator else None

    x = _center_columns(left)
    y = _center_columns(right)
    # Dual formulation avoids multiplying every feature pair when observations are few.
    gram_x = _gram(x)
    gram_y = _gram(y)
    numerator = _frobenius_inner(gram_x, gram_y)
    denominator = math.sqrt(_frobenius_inner(gram_x, gram_x) * _frobenius_inner(gram_y, gram_y))
    return numerator / denominator if denominator else None


def compare_cross_model(outputs: dict, sequence_order: list[str], spec: CrossModelSpec) -> dict:
    if not spec.enabled:
        return {"enabled": False}
    if len(outputs) < 2:
        return {
            "enabled": True,
            "available": False,
            "reason": "At least two successful models are required",
            "model_order": list(outputs),
            "comparisons": {},
            "warnings": [],
        }
    if spec.representation not in {
        RepresentationMethod.MEAN,
        RepresentationMethod.MAX,
        RepresentationMethod.NATIVE,
    }:
        raise ValidationError(
            "Cross-model sequence relationships require mean, max, or native representation"
        )
    if any(k <= 0 for k in spec.neighbor_k):
        raise ValidationError("cross_model.neighbor_k values must be positive")
    if spec.max_disagreements <= 0:
        raise ValidationError("cross_model.max_disagreements must be positive")

    methods = set(spec.methods)
    warnings: list[str] = []
    vector_maps = {
        model_id: _representation_map(output, str(spec.representation))
        for model_id, output in outputs.items()
    }
    vector_maps = {
        model_id: mapping
        for model_id, mapping in vector_maps.items()
        if all(sequence_id in mapping for sequence_id in sequence_order)
    }
    model_order = list(vector_maps)
    relationships = {
        model_id: pairwise([mapping[item] for item in sequence_order], "cosine_similarity")
        for model_id, mapping in vector_maps.items()
    }
    comparisons: dict = {}

    if methods & SEQUENCE_RELATIONSHIP_METHODS and len(model_order) < 2:
        warnings.append(
            f"Fewer than two models produced complete {spec.representation} representations"
        )
    if CrossModelMethod.MATRIX_SPEARMAN in methods:
        comparisons["matrix_spearman"] = _matrix_correlation(
            relationships, model_order, method="spearman"
        )
    if CrossModelMethod.MATRIX_PEARSON in methods:
        comparisons["matrix_pearson"] = _matrix_correlation(
            relationships, model_order, method="pearson"
        )
    if CrossModelMethod.LINEAR_CKA in methods:
        comparisons["linear_cka"] = _cka_models(vector_maps, sequence_order, model_order)
        if len(sequence_order) == 2:
            warnings.append("Sequence-level CKA with only two sequences is usually uninformative")
    if CrossModelMethod.NEIGHBOR_OVERLAP in methods:
        comparisons["neighbor_overlap"] = _neighbor_overlap(
            relationships, model_order, spec.neighbor_k
        )
    if CrossModelMethod.CONSENSUS_RANK in methods:
        comparisons["consensus_rank"] = _consensus_rank(relationships, model_order, sequence_order)
    if CrossModelMethod.PAIR_DISAGREEMENT in methods:
        comparisons["pair_disagreement"] = _pair_disagreement(
            relationships, model_order, sequence_order, spec.max_disagreements
        )
    if methods & RESIDUE_METHODS:
        residue = _compare_residues(outputs, sequence_order, methods)
        comparisons.update(residue)

    overall_available = any(
        comparison.get("available", False)
        for comparison in comparisons.values()
        if isinstance(comparison, dict)
    )
    return {
        "enabled": True,
        "available": overall_available,
        "model_order": list(outputs),
        "relationship_model_order": model_order,
        "representation": str(spec.representation),
        "relationship_metric": "within-model cosine similarity",
        "relationship_matrices": relationships,
        "comparisons": comparisons,
        "warnings": warnings,
        "scientific_guardrail": "No raw vectors from different PLMs were directly compared",
    }


def _representation_map(output: dict, method: str) -> dict[str, list]:
    result = {}
    for item in output.get("representations", []):
        if item["method"] == method and item["sequence_id"] not in result:
            result[item["sequence_id"]] = item["values"]
    return result


def _matrix_correlation(relationships: dict, model_order: list[str], *, method: str) -> dict:
    if len(model_order) < 2:
        return _unavailable("At least two complete model relationship matrices are required")
    sequence_count = len(next(iter(relationships.values())))
    if sequence_count < 3:
        return _unavailable("Matrix correlation requires at least three sequences")
    matrix = _identity_matrix(len(model_order))
    for i in range(len(model_order)):
        for j in range(i + 1, len(model_order)):
            left, right = relationships[model_order[i]], relationships[model_order[j]]
            value = (
                matrix_spearman(left, right)
                if method == "spearman"
                else pearson(upper_triangle(left), upper_triangle(right))
            )
            matrix[i][j] = matrix[j][i] = value
    return {"available": True, "model_order": model_order, "values": matrix}


def _cka_models(vector_maps: dict, sequence_order: list[str], model_order: list[str]) -> dict:
    if len(model_order) < 2:
        return _unavailable("At least two complete model representations are required")
    if len(sequence_order) < 2:
        return _unavailable("Sequence-level CKA requires at least two sequences")
    matrix = _identity_matrix(len(model_order))
    for i in range(len(model_order)):
        for j in range(i + 1, len(model_order)):
            left = [vector_maps[model_order[i]][item] for item in sequence_order]
            right = [vector_maps[model_order[j]][item] for item in sequence_order]
            value = linear_cka(left, right)
            matrix[i][j] = matrix[j][i] = value
    return {"available": True, "model_order": model_order, "values": matrix}


def _neighbor_overlap(
    relationships: dict, model_order: list[str], requested_k: tuple[int, ...]
) -> dict:
    if len(model_order) < 2:
        return _unavailable("At least two model relationship matrices are required")
    n = len(next(iter(relationships.values())))
    valid_k = sorted({min(k, n - 1) for k in requested_k if n > 1})
    if not valid_k:
        return _unavailable("Neighbor overlap requires at least two sequences")
    by_k = {}
    for k in valid_k:
        neighbors = {
            model_id: [
                set(
                    sorted(
                        (j for j in range(n) if j != i),
                        key=lambda j: relationships[model_id][i][j],
                        reverse=True,
                    )[:k]
                )
                for i in range(n)
            ]
            for model_id in model_order
        }
        matrix = _identity_matrix(len(model_order))
        for i in range(len(model_order)):
            for j in range(i + 1, len(model_order)):
                scores = []
                for sequence_index in range(n):
                    left = neighbors[model_order[i]][sequence_index]
                    right = neighbors[model_order[j]][sequence_index]
                    scores.append(len(left & right) / len(left | right))
                matrix[i][j] = matrix[j][i] = sum(scores) / len(scores)
        by_k[str(k)] = matrix
    return {"available": True, "model_order": model_order, "by_k": by_k}


def _consensus_rank(relationships: dict, model_order: list[str], sequence_order: list[str]) -> dict:
    if len(model_order) < 2:
        return _unavailable("At least two model relationship matrices are required")
    pair_count = len(sequence_order) * (len(sequence_order) - 1) // 2
    if pair_count < 2:
        return _unavailable("Rank consensus requires at least three sequences")
    profiles = _rank_profiles(relationships, model_order)
    consensus = [
        sum(profile[index] for profile in profiles.values()) / len(profiles)
        for index in range(pair_count)
    ]
    matrix = _pair_values_to_matrix(consensus, len(sequence_order), diagonal=1.0)
    return {
        "available": True,
        "sequence_order": sequence_order,
        "values": matrix,
        "scale": "mean within-model pair-rank percentile; 1 is most similar",
    }


def _pair_disagreement(
    relationships: dict,
    model_order: list[str],
    sequence_order: list[str],
    limit: int,
) -> dict:
    if len(model_order) < 2:
        return _unavailable("At least two model relationship matrices are required")
    pair_count = len(sequence_order) * (len(sequence_order) - 1) // 2
    if pair_count < 2:
        return _unavailable("Rank disagreement requires at least three sequences")
    profiles = _rank_profiles(relationships, model_order)
    pairs = []
    index = 0
    disagreement_values = []
    for i in range(len(sequence_order)):
        for j in range(i + 1, len(sequence_order)):
            model_ranks = {model: profiles[model][index] for model in model_order}
            disagreement = max(model_ranks.values()) - min(model_ranks.values())
            disagreement_values.append(disagreement)
            pairs.append(
                {
                    "left": sequence_order[i],
                    "right": sequence_order[j],
                    "rank_range": disagreement,
                    "model_rank_percentiles": model_ranks,
                }
            )
            index += 1
    pairs.sort(key=lambda item: item["rank_range"], reverse=True)
    return {
        "available": True,
        "top_pairs": pairs[:limit],
        "values": _pair_values_to_matrix(disagreement_values, len(sequence_order), diagonal=0.0),
        "scale": "range of within-model pair-rank percentiles",
    }


def _compare_residues(outputs: dict, sequence_order: list[str], methods: set) -> dict:
    residue_maps = {
        model_id: _representation_map(output, "residue") for model_id, output in outputs.items()
    }
    result: dict[str, dict] = {}
    for method in methods & RESIDUE_METHODS:
        by_sequence = {}
        for sequence_id in sequence_order:
            model_order = [model for model, values in residue_maps.items() if sequence_id in values]
            if len(model_order) < 2:
                by_sequence[sequence_id] = _unavailable(
                    "At least two models must produce residue representations"
                )
                continue
            residue_count = len(residue_maps[model_order[0]][sequence_id])
            if any(len(residue_maps[model][sequence_id]) != residue_count for model in model_order):
                by_sequence[sequence_id] = _unavailable(
                    "Residue mappings have different observation counts"
                )
                continue
            if method == CrossModelMethod.RESIDUE_RSA and residue_count < 3:
                by_sequence[sequence_id] = _unavailable(
                    "Residue RSA requires at least three residues"
                )
                continue
            if method == CrossModelMethod.RESIDUE_CKA and residue_count < 2:
                by_sequence[sequence_id] = _unavailable(
                    "Residue CKA requires at least two residues"
                )
                continue
            matrix = _identity_matrix(len(model_order))
            self_similarity = {}
            if method == CrossModelMethod.RESIDUE_RSA:
                self_similarity = {
                    model: pairwise(residue_maps[model][sequence_id], "cosine_similarity")
                    for model in model_order
                }
            for i in range(len(model_order)):
                for j in range(i + 1, len(model_order)):
                    if method == CrossModelMethod.RESIDUE_CKA:
                        value = linear_cka(
                            residue_maps[model_order[i]][sequence_id],
                            residue_maps[model_order[j]][sequence_id],
                        )
                    else:
                        value = matrix_spearman(
                            self_similarity[model_order[i]], self_similarity[model_order[j]]
                        )
                    matrix[i][j] = matrix[j][i] = value
            by_sequence[sequence_id] = {
                "available": True,
                "model_order": model_order,
                "residue_count": residue_count,
                "values": matrix,
            }
        result[str(method)] = {
            "available": any(item.get("available", False) for item in by_sequence.values()),
            "by_sequence": by_sequence,
        }
    return result


def _rank_profiles(relationships: dict, model_order: list[str]) -> dict[str, list[float]]:
    return {
        model: _descending_percentiles(upper_triangle(relationships[model]))
        for model in model_order
    }


def _descending_percentiles(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    result = [0.0] * len(values)
    index = 0
    denominator = max(1, len(values) - 1)
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        percentile = 1.0 - ((index + end - 1) / 2) / denominator
        for original_index in order[index:end]:
            result[original_index] = percentile
        index = end
    return result


def _identity_matrix(size: int) -> list[list[float | None]]:
    return [[1.0 if i == j else None for j in range(size)] for i in range(size)]


def _pair_values_to_matrix(values: list[float], size: int, *, diagonal: float) -> list[list[float]]:
    matrix = [[diagonal if i == j else 0.0 for j in range(size)] for i in range(size)]
    index = 0
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i][j] = matrix[j][i] = values[index]
            index += 1
    return matrix


def _center_columns(rows: list[list[float]]) -> list[list[float]]:
    width = len(rows[0])
    means = [sum(row[column] for row in rows) / len(rows) for column in range(width)]
    return [[value - means[column] for column, value in enumerate(row)] for row in rows]


def _gram(rows: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a * b for a, b in zip(left, right, strict=True)) for right in rows] for left in rows
    ]


def _frobenius_inner(left: list[list[float]], right: list[list[float]]) -> float:
    return sum(
        a * b
        for row_a, row_b in zip(left, right, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}
