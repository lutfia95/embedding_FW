from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any

from . import __version__
from .errors import ResourceLimitError, ValidationError
from .models import ImportedNormalization, RepresentationMethod

_SAFE_SPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ALLOWED_METHODS = {
    RepresentationMethod.MEAN,
    RepresentationMethod.MAX,
    RepresentationMethod.NATIVE,
    RepresentationMethod.RESIDUE,
}


def build_imported_space(
    *,
    space_name: str,
    representations: list[dict[str, Any]],
    source: dict[str, Any] | None = None,
    max_records: int = 10_000,
    max_dimension: int = 32_768,
    max_scalar_values: int = 5_000_000,
) -> dict[str, Any]:
    """Validate precomputed embeddings and expose them as one model-like result.

    Imported values are structurally validated but are never described as a
    conformance-tested model output. The returned result deliberately matches
    the model/result contract consumed by cross-model comparisons.
    """
    if not isinstance(space_name, str) or not _SAFE_SPACE_NAME.fullmatch(space_name):
        raise ValidationError(
            "space_name must contain 1-128 letters, digits, dots, underscores, colons, or hyphens"
        )
    if not representations:
        raise ValidationError("At least one imported representation is required")
    if max_records < 1 or max_dimension < 1 or max_scalar_values < 1:
        raise ValueError("Import safety limits must be positive")

    source = dict(source or {})
    sequence_order: list[str] | None = None
    materialized: list[dict[str, Any]] = []
    representation_summary = []
    total_values = 0
    dimensions: set[int] = set()
    seen_representation_keys: set[tuple[str, str]] = set()

    for representation in representations:
        try:
            method = RepresentationMethod(representation.get("method", "mean"))
            normalization = ImportedNormalization(
                representation.get("normalization", ImportedNormalization.UNKNOWN)
            )
        except ValueError as exc:
            raise ValidationError(f"Invalid imported representation: {exc}") from exc
        if method not in _ALLOWED_METHODS:
            raise ValidationError(
                "Imported representations currently support mean, max, native, or residue",
                context={"method": str(method)},
            )
        layer = representation.get("layer", "external")
        key = (str(method), str(layer))
        if key in seen_representation_keys:
            raise ValidationError(
                "Each imported method/layer combination must be unique",
                context={"method": str(method), "layer": layer},
            )
        seen_representation_keys.add(key)

        records = representation.get("embeddings")
        if not isinstance(records, list) or not records:
            raise ValidationError(
                "Each imported representation requires a non-empty embeddings list",
                context={"method": str(method), "layer": layer},
            )
        if len(records) > max_records:
            raise ResourceLimitError(
                f"Imported representation exceeds the {max_records} record limit",
                context={"records": len(records), "limit": max_records},
            )

        ids: list[str] = []
        seen_ids: set[str] = set()
        expected_dimension: int | None = None
        residue_counts: list[int] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValidationError(
                    "Imported embedding records must be objects",
                    context={"record_index": record_index},
                )
            sequence_id = record.get("sequence_id")
            if not isinstance(sequence_id, str) or not sequence_id.strip():
                raise ValidationError(
                    "Imported sequence_id must be a non-empty string",
                    context={"record_index": record_index},
                )
            if len(sequence_id) > 256:
                raise ValidationError(
                    "Imported sequence_id exceeds 256 characters",
                    context={"record_index": record_index},
                )
            if sequence_id in seen_ids:
                raise ValidationError(
                    f"Duplicate imported sequence ID: {sequence_id}",
                    context={"record_index": record_index},
                )
            seen_ids.add(sequence_id)
            ids.append(sequence_id)

            values = record.get("values")
            if method == RepresentationMethod.RESIDUE:
                cleaned, dimension = _validate_matrix(
                    values,
                    max_dimension=max_dimension,
                    context={"sequence_id": sequence_id, "method": str(method)},
                )
                residue_counts.append(len(cleaned))
                scalar_count = len(cleaned) * dimension
                shape = [len(cleaned), dimension]
                norm_before = None
            else:
                cleaned = _validate_vector(
                    values,
                    max_dimension=max_dimension,
                    context={"sequence_id": sequence_id, "method": str(method)},
                )
                dimension = len(cleaned)
                scalar_count = dimension
                shape = [dimension]
                norm_before = math.sqrt(sum(value * value for value in cleaned))
            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise ValidationError(
                    "Embedding dimensions must be consistent within a representation",
                    context={
                        "sequence_id": sequence_id,
                        "expected_dimension": expected_dimension,
                        "actual_dimension": dimension,
                    },
                )
            total_values += scalar_count
            if total_values > max_scalar_values:
                raise ResourceLimitError(
                    f"Import exceeds the {max_scalar_values} scalar-value limit",
                    context={"scalar_values": total_values, "limit": max_scalar_values},
                )
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValidationError(
                    "Imported embedding metadata must be an object",
                    context={"record_index": record_index, "sequence_id": sequence_id},
                )
            materialized.append(
                {
                    "sequence_id": sequence_id,
                    "method": str(method),
                    "layer": layer,
                    "values": cleaned,
                    "shape": shape,
                    "norm_before": norm_before,
                    "normalization": str(normalization),
                    "metadata": {
                        **metadata,
                        "imported": True,
                    },
                }
            )

        if sequence_order is None:
            sequence_order = ids
        elif ids != sequence_order:
            raise ValidationError(
                "Every imported representation must contain identical sequence IDs in identical order",
                context={"method": str(method), "layer": layer},
            )
        dimensions.add(expected_dimension or 0)
        representation_summary.append(
            {
                "method": str(method),
                "layer": layer,
                "normalization": str(normalization),
                "sequence_count": len(records),
                "embedding_dimension": expected_dimension,
                "residue_count_min": min(residue_counts) if residue_counts else None,
                "residue_count_max": max(residue_counts) if residue_counts else None,
            }
        )

    sequence_order = sequence_order or []
    checksum_payload = {
        "space_name": space_name,
        "source": source,
        "representations": representations,
    }
    try:
        canonical = json.dumps(
            checksum_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Imported metadata must be finite JSON data: {exc}") from exc
    payload_sha256 = hashlib.sha256(canonical).hexdigest()
    imported_at = datetime.now(UTC).isoformat()
    methods = sorted({item["method"] for item in representation_summary})
    manifest = {
        "id": space_name,
        "display_name": source.get("display_name", space_name),
        "family": source.get("family", "external"),
        "provider": source.get("provider", "external"),
        "artifact": source.get("artifact", "client-supplied"),
        "revision": source.get("revision", "unreported"),
        "license": source.get("license", "unreported"),
        "citation": source.get("citation", "unreported"),
        "architecture": source.get("architecture", "external/precomputed"),
        "input_modalities": ["precomputed_embedding"],
        "alphabet": "not_available",
        "ambiguous_residues": "not_available",
        "tokenizer_convention": source.get("tokenizer_convention", "unreported"),
        "max_native_length": None,
        "num_layers": None,
        "embedding_dimension": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "representations": methods,
        "default_layer": representation_summary[0]["layer"],
        "one_token_per_residue": source.get("one_token_per_residue"),
        "special_tokens": source.get("special_tokens", "unreported"),
        "devices": [],
        "precisions": [],
        "parameters": source.get("parameters"),
        "adapter_version": "external-import-v1",
        "conformance": "externally_supplied_structural_validation_only",
        "limitations": [
            "Values were supplied by the client and were not reproduced from the source model",
            "Structural validation does not establish biological correctness or model conformance",
        ],
        "aliases": [],
        "backend": "external_import",
        "preprocess": source.get("preprocess", "unreported"),
        "trust_remote_code": source.get("trust_remote_code"),
    }
    provenance = {
        "framework_version": __version__,
        "schema_version": "1.0",
        "operation": "external_space_import",
        "imported_at": imported_at,
        "space_name": space_name,
        "payload_sha256": payload_sha256,
        "sequence_count": len(sequence_order),
        "scalar_value_count": total_values,
        "representations": representation_summary,
        "source": source,
        "validation_scope": "structure, dimensions, finite numeric values, IDs, ordering, and limits",
        "conformance_scope": "not evaluated",
    }
    return {
        "schema_version": "1.0",
        "sequence_order": sequence_order,
        "models": {
            space_name: {
                "manifest": manifest,
                "availability": {"available": True, "reason_code": None, "detail": None},
                "representations": materialized,
                "matrices": {},
                "residue_pairs": {},
                "provenance": provenance,
            }
        },
        "failures": [],
        "requested_visualizations": [],
        "provenance": provenance,
    }


def _validate_vector(values: Any, *, max_dimension: int, context: dict[str, Any]) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValidationError("Imported vector must be a non-empty list", context=context)
    if len(values) > max_dimension:
        raise ResourceLimitError(
            f"Embedding dimension exceeds the {max_dimension} limit",
            context={**context, "dimension": len(values), "limit": max_dimension},
        )
    return [_finite_number(value, context=context) for value in values]


def _validate_matrix(
    values: Any, *, max_dimension: int, context: dict[str, Any]
) -> tuple[list[list[float]], int]:
    if not isinstance(values, list) or not values:
        raise ValidationError("Imported residue matrix must be a non-empty list", context=context)
    rows = []
    dimension = None
    for row_index, row in enumerate(values):
        row_context = {**context, "residue_index": row_index}
        cleaned = _validate_vector(row, max_dimension=max_dimension, context=row_context)
        if dimension is None:
            dimension = len(cleaned)
        elif len(cleaned) != dimension:
            raise ValidationError(
                "All rows in an imported residue matrix must have the same dimension",
                context={**row_context, "expected_dimension": dimension},
            )
        rows.append(cleaned)
    return rows, dimension or 0


def _finite_number(value: Any, *, context: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("Embedding values must be numbers", context=context)
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError("Embedding values must be finite", context=context)
    return result
