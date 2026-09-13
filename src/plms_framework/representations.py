from __future__ import annotations

import math

from .errors import CapabilityError, ValidationError
from .models import (
    Normalization,
    ProteinSequence,
    RawEmbedding,
    RepresentationMethod,
    RepresentationResult,
    RepresentationSpec,
)


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: list[float], method: Normalization) -> tuple[list[float], float]:
    norm = l2_norm(vector)
    if method == Normalization.NONE:
        return list(vector), norm
    if norm == 0:
        return list(vector), norm
    return [value / norm for value in vector], norm


def mean_pool(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValidationError("Cannot pool an empty residue representation")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValidationError("Residue embedding rows have inconsistent dimensions")
    return [sum(row[column] for row in rows) / len(rows) for column in range(width)]


def max_pool(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValidationError("Cannot pool an empty residue representation")
    return [max(row[column] for row in rows) for column in range(len(rows[0]))]


def materialize(
    raw: RawEmbedding, sequence: ProteinSequence, spec: RepresentationSpec
) -> RepresentationResult:
    method = spec.method
    if method == RepresentationMethod.RESIDUE:
        values = []
        norms = []
        for row in raw.residues:
            normalized, norm = normalize(row, spec.normalize)
            values.append(normalized)
            norms.append(norm)
        return RepresentationResult(
            raw.sequence_id,
            str(method),
            spec.layer,
            values,
            (len(values), len(values[0]) if values else 0),
            None,
            str(spec.normalize),
            {"row_norms_before": norms},
        )
    if method == RepresentationMethod.MEAN:
        vector = mean_pool(raw.residues)
    elif method == RepresentationMethod.MAX:
        vector = max_pool(raw.residues)
    elif method == RepresentationMethod.NATIVE:
        if raw.native is None:
            raise CapabilityError("Adapter did not return a native sequence representation")
        vector = raw.native
    elif method == RepresentationMethod.REGION_MEAN:
        if not sequence.regions:
            raise ValidationError(f"Sequence {sequence.id!r} has no regions for region_mean")
        region_vectors = []
        for region in sequence.regions:
            vector, norm = normalize(
                mean_pool(raw.residues[region.start - 1 : region.end]), spec.normalize
            )
            region_vectors.append(
                {
                    "name": region.name,
                    "start": region.start,
                    "end": region.end,
                    "values": vector,
                    "norm_before": norm,
                }
            )
        width = len(region_vectors[0]["values"])
        return RepresentationResult(
            raw.sequence_id,
            str(method),
            spec.layer,
            region_vectors,
            (len(region_vectors), width),
            None,
            str(spec.normalize),
        )
    elif method == RepresentationMethod.WINDOW_MEAN:
        size = spec.window_size
        step = spec.window_step or size
        if size is None or size <= 0 or step is None or step <= 0:
            raise ValidationError("window_mean requires positive window_size and window_step")
        windows = []
        for start in range(0, len(raw.residues), step):
            rows = raw.residues[start : start + size]
            if not rows:
                continue
            vector, norm = normalize(mean_pool(rows), spec.normalize)
            windows.append(
                {
                    "start": start + 1,
                    "end": start + len(rows),
                    "values": vector,
                    "norm_before": norm,
                }
            )
            if start + size >= len(raw.residues):
                break
        width = len(windows[0]["values"])
        return RepresentationResult(
            raw.sequence_id,
            str(method),
            spec.layer,
            windows,
            (len(windows), width),
            None,
            str(spec.normalize),
        )
    else:
        raise CapabilityError(f"Representation method is not implemented: {method}")
    vector, norm = normalize(vector, spec.normalize)
    return RepresentationResult(
        raw.sequence_id,
        str(method),
        spec.layer,
        vector,
        (len(vector),),
        norm,
        str(spec.normalize),
    )
