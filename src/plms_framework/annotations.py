from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from .errors import ResourceLimitError, ValidationError
from .metrics import pairwise
from .models import AnnotationAnalysisSpec, MissingAnnotationPolicy, RepresentationMethod

_VECTOR_METHODS = {
    RepresentationMethod.MEAN,
    RepresentationMethod.MAX,
    RepresentationMethod.NATIVE,
}


def analyze_categorical_annotations(
    outputs: dict[str, dict[str, Any]],
    sequence_order: list[str],
    annotations: list[dict[str, Any]],
    spec: AnnotationAnalysisSpec,
) -> dict[str, Any]:
    """Measure categorical-label organization in within-model neighborhoods."""
    if not spec.feature.strip():
        raise ValidationError("A non-empty categorical feature name is required")
    if spec.representation not in _VECTOR_METHODS:
        raise ValidationError("Annotation analysis requires mean, max, or native vectors")
    if not spec.k or any(k < 1 for k in spec.k):
        raise ValidationError("Every requested k must be positive")
    if len(sequence_order) != len(set(sequence_order)):
        raise ValidationError("Source sequence IDs must be unique")

    annotation_map: dict[str, dict[str, Any]] = {}
    known_ids = set(sequence_order)
    for index, record in enumerate(annotations):
        if not isinstance(record, dict):
            raise ValidationError(
                "Annotation records must be objects", context={"record_index": index}
            )
        sequence_id = record.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValidationError(
                "Annotation sequence_id must be a non-empty string",
                context={"record_index": index},
            )
        if sequence_id not in known_ids:
            raise ValidationError(
                "Annotation contains an ID absent from the source job",
                context={"sequence_id": sequence_id},
            )
        if sequence_id in annotation_map:
            raise ValidationError(
                f"Duplicate annotation sequence ID: {sequence_id}",
                context={"record_index": index},
            )
        categorical = record.get("categorical") or {}
        if not isinstance(categorical, dict):
            raise ValidationError(
                "categorical annotations must be an object",
                context={"sequence_id": sequence_id},
            )
        annotation_map[sequence_id] = categorical

    missing_ids = [
        sequence_id
        for sequence_id in sequence_order
        if sequence_id not in annotation_map
        or spec.feature not in annotation_map[sequence_id]
        or annotation_map[sequence_id][spec.feature] is None
    ]
    if missing_ids and spec.missing_policy == MissingAnnotationPolicy.ERROR:
        raise ValidationError(
            f"Missing categorical feature {spec.feature!r} for source sequences",
            context={"missing_sequence_ids": missing_ids[:100], "missing_count": len(missing_ids)},
        )
    analyzed_ids = [sequence_id for sequence_id in sequence_order if sequence_id not in missing_ids]
    if len(analyzed_ids) < 2:
        raise ValidationError("Annotation analysis requires at least two labeled sequences")
    if len(analyzed_ids) > spec.max_exact_sequences:
        raise ResourceLimitError(
            f"Exact annotation analysis is limited to {spec.max_exact_sequences} sequences",
            context={"sequence_count": len(analyzed_ids)},
        )

    raw_labels = [annotation_map[sequence_id][spec.feature] for sequence_id in analyzed_ids]
    labels = [_label_key(value) for value in raw_labels]
    display_by_key = {key: value for key, value in zip(labels, raw_labels, strict=True)}
    class_keys = sorted(set(labels))
    if len(class_keys) < 2:
        raise ValidationError("Categorical annotation analysis requires at least two classes")
    class_sizes = {key: labels.count(key) for key in class_keys}
    n = len(analyzed_ids)
    valid_k = sorted({min(k, n - 1) for k in spec.k})
    warnings = []
    for requested in sorted(set(spec.k)):
        if requested >= n:
            warnings.append(f"k={requested} was reduced to {n - 1} for {n} labeled sequences")
    singleton_classes = [display_by_key[key] for key, size in class_sizes.items() if size == 1]
    if singleton_classes:
        warnings.append(
            "Singleton classes have no same-class retrieval target and receive null recall"
        )
    smallest_class = min(class_sizes.values())
    for k in valid_k:
        if k >= smallest_class:
            warnings.append(
                f"k={k} is at least the smallest class size ({smallest_class}); "
                "interpret class comparisons with the reported class sizes"
            )

    canonical_annotations = [
        {"sequence_id": sequence_id, "value": annotation_map[sequence_id][spec.feature]}
        for sequence_id in analyzed_ids
    ]
    annotations_sha256 = hashlib.sha256(
        json.dumps(
            canonical_annotations,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    model_results = {}
    for model_id, output in outputs.items():
        selected = _select_vectors(output, analyzed_ids, spec)
        if selected is None:
            model_results[model_id] = {
                "available": False,
                "reason": f"Complete {spec.representation} vectors were not found",
            }
            continue
        layer, vectors, selection_warnings = selected
        relationships = pairwise(vectors, "cosine_similarity")
        by_k = {}
        for k in valid_k:
            by_k[str(k)] = _analyze_k(
                relationships,
                analyzed_ids,
                labels,
                class_keys,
                display_by_key,
                class_sizes,
                k,
            )
        model_results[model_id] = {
            "available": True,
            "layer": layer,
            "sequence_count": n,
            "embedding_dimension": len(vectors[0]),
            "metric": "cosine_similarity",
            "by_k": by_k,
            "warnings": selection_warnings,
        }

    return {
        "available": any(result.get("available", False) for result in model_results.values()),
        "feature": spec.feature,
        "representation": str(spec.representation),
        "requested_layer": spec.layer,
        "missing_policy": str(spec.missing_policy),
        "source_sequence_count": len(sequence_order),
        "analyzed_sequence_count": n,
        "dropped_sequence_ids": missing_ids,
        "annotations_sha256": annotations_sha256,
        "class_summary": [
            {"label": display_by_key[key], "count": class_sizes[key]} for key in class_keys
        ],
        "models": model_results,
        "warnings": warnings,
        "definitions": {
            "micro_consistency": "fraction of all selected neighbors sharing the query label",
            "macro_consistency": "unweighted mean of per-class consistency values",
            "chance_baseline": (
                "expected same-label probability when drawing another sequence from the "
                "observed labeled collection"
            ),
            "chance_adjusted_consistency": "(micro_consistency - chance) / (1 - chance)",
            "recall_at_k": "same-class neighbors retrieved divided by all available same-class targets",
            "class_mixing": "row-normalized counts from query class to neighbor class",
        },
        "scientific_guardrail": (
            "Annotation agreement measures organization relative to supplied labels; it does not "
            "establish label correctness, causal biology, or generalization to held-out families"
        ),
    }


def _analyze_k(
    relationships: list[list[float]],
    sequence_ids: list[str],
    labels: list[str],
    class_keys: list[str],
    display_by_key: dict[str, Any],
    class_sizes: dict[str, int],
    k: int,
) -> dict[str, Any]:
    class_index = {key: index for index, key in enumerate(class_keys)}
    mixing_counts = [[0 for _ in class_keys] for _ in class_keys]
    per_sample = []
    consistency_by_class: dict[str, list[float]] = defaultdict(list)
    recall_by_class: dict[str, list[float]] = defaultdict(list)
    total_matches = 0

    for index, row in enumerate(relationships):
        neighbors = sorted(
            (candidate for candidate in range(len(row)) if candidate != index),
            key=lambda candidate: (-row[candidate], candidate),
        )[:k]
        matches = sum(labels[neighbor] == labels[index] for neighbor in neighbors)
        consistency = matches / k
        available_targets = class_sizes[labels[index]] - 1
        recall = matches / available_targets if available_targets else None
        total_matches += matches
        consistency_by_class[labels[index]].append(consistency)
        if recall is not None:
            recall_by_class[labels[index]].append(recall)
        source_class_index = class_index[labels[index]]
        for neighbor in neighbors:
            mixing_counts[source_class_index][class_index[labels[neighbor]]] += 1
        per_sample.append(
            {
                "sequence_id": sequence_ids[index],
                "label": display_by_key[labels[index]],
                "same_class_neighbors": matches,
                "consistency": consistency,
                "recall": recall,
                "neighbor_ids": [sequence_ids[neighbor] for neighbor in neighbors],
            }
        )

    micro = total_matches / (len(sequence_ids) * k)
    per_class = []
    class_means = []
    for key in class_keys:
        consistency_values = consistency_by_class[key]
        mean_consistency = sum(consistency_values) / len(consistency_values)
        class_means.append(mean_consistency)
        recalls = recall_by_class.get(key, [])
        per_class.append(
            {
                "label": display_by_key[key],
                "count": class_sizes[key],
                "mean_consistency": mean_consistency,
                "mean_recall": sum(recalls) / len(recalls) if recalls else None,
            }
        )
    chance = sum(size * (size - 1) for size in class_sizes.values()) / (
        len(sequence_ids) * (len(sequence_ids) - 1)
    )
    chance_adjusted = (micro - chance) / (1.0 - chance) if chance < 1.0 else None
    mixing_fractions = []
    for row in mixing_counts:
        total = sum(row)
        mixing_fractions.append([value / total if total else 0.0 for value in row])
    return {
        "k": k,
        "micro_consistency": micro,
        "macro_consistency": sum(class_means) / len(class_means),
        "chance_baseline": chance,
        "chance_adjusted_consistency": chance_adjusted,
        "precision_at_k": micro,
        "mean_recall_at_k": _mean_defined(
            [item["recall"] for item in per_sample if item["recall"] is not None]
        ),
        "per_class": per_class,
        "per_sample": per_sample,
        "class_mixing": {
            "class_order": [display_by_key[key] for key in class_keys],
            "counts": mixing_counts,
            "row_fractions": mixing_fractions,
        },
    }


def _select_vectors(
    output: dict[str, Any], sequence_ids: list[str], spec: AnnotationAnalysisSpec
) -> tuple[str | int, list[list[float]], list[str]] | None:
    groups: dict[tuple[str, str | int], dict[str, Any]] = {}
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
        if all(sequence_id in mapping for sequence_id in sequence_ids)
    ]
    if not complete:
        return None
    warnings = []
    if spec.layer is None and len(complete) > 1:
        warnings.append(
            f"Multiple complete layers were present; selected the first layer {complete[0][0][1]}"
        )
    (_, layer), mapping = complete[0]
    vectors = [_clean_vector(mapping[sequence_id], sequence_id) for sequence_id in sequence_ids]
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValidationError("Annotation-analysis vectors must have a consistent dimension")
    return layer, vectors, warnings


def _clean_vector(values: Any, sequence_id: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValidationError(
            "Annotation analysis requires non-empty vectors",
            context={"sequence_id": sequence_id},
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                "Annotation-analysis vectors must contain numbers",
                context={"sequence_id": sequence_id},
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError(
                "Annotation-analysis vectors must contain finite values",
                context={"sequence_id": sequence_id},
            )
        result.append(number)
    return result


def _label_key(value: Any) -> str:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, str):
        kind = "str"
    else:
        raise ValidationError("Categorical values must be strings, integers, booleans, or null")
    return f"{kind}:{json.dumps(value, ensure_ascii=False, sort_keys=True)}"


def _mean_defined(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
