#!/usr/bin/env python3
"""Score noisy protein FASTA sequences against one reference sequence.

Each FASTA file in ``NOISY_FASTA_DIR`` must contain exactly one record whose
full header matches the selected reference header.  All sequences are embedded
in one batch and the reference-to-noisy scores are written to one CSV file.

Example:

    python3.12 noisy_seq_analyser.py reference.fasta noisy_fastas \
        --output noisy_sequence_scores.csv

The default model and metric choices mirror the pairwise scores demonstrated by
``python_ex.py``.  The first run may download the ESM-2 and ESM-C checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plms_framework import Pipeline
from plms_framework.errors import FrameworkError, ValidationError
from plms_framework.metrics import (
    SEQUENCE_METRICS,
    VECTOR_METRICS,
    align_similarity,
    residue_similarity,
    sinkhorn_similarity,
    symmetric_best_match,
)
from plms_framework.models import ProteinSequence
from plms_framework.sequences import parse_fasta
from plms_framework.serialization import request_from_dict


DEFAULT_MODELS = ("esm2-8m", "esmc-300m")
FASTA_SUFFIXES = frozenset({".fa", ".faa", ".fas", ".fasta", ".fsa"})
SEQUENCE_METRIC_NAMES = (
    "length_ratio",
    "sequence_identity",
    "kmer_jaccard",
    "blosum62_global",
    "blosum62_local",
)
VECTOR_METRIC_NAMES = (
    "cosine_similarity",
    "cosine_distance",
    "angular_distance",
    "correlation_distance",
    "euclidean_distance",
)


@dataclass(frozen=True)
class NoisySequence:
    """A selected noisy FASTA record and its safe framework identifier."""

    path: Path
    internal_id: str
    sequence: str


def read_fasta(path: Path) -> tuple[ProteinSequence, ...]:
    """Read one FASTA file and add its path to parsing errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read FASTA file {path}: {exc}") from exc
    try:
        return parse_fasta(text)
    except FrameworkError as exc:
        raise ValidationError(f"Invalid FASTA file {path}: {exc}") from exc


def select_reference(path: Path, requested_header: str | None) -> ProteinSequence:
    records = read_fasta(path)
    if requested_header is None:
        if len(records) != 1:
            raise ValidationError(
                f"Reference FASTA {path} contains {len(records)} records; "
                "use --header to select one by its full FASTA header"
            )
        return records[0]

    matches = [record for record in records if record.id == requested_header]
    if len(matches) != 1:
        raise ValidationError(
            f"Expected exactly one reference record with header {requested_header!r} "
            f"in {path}, found {len(matches)}"
        )
    return matches[0]


def discover_noisy_fastas(directory: Path, reference_path: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValidationError(f"Noisy FASTA directory does not exist: {directory}")
    reference_resolved = reference_path.resolve()
    paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in FASTA_SUFFIXES
            and path.resolve() != reference_resolved
        ),
        key=lambda path: path.name,
    )
    if not paths:
        suffixes = ", ".join(sorted(FASTA_SUFFIXES))
        raise ValidationError(
            f"No FASTA files found in {directory} (recognized suffixes: {suffixes})"
        )
    return paths


def load_noisy_sequences(paths: list[Path], header: str) -> list[NoisySequence]:
    noisy_sequences: list[NoisySequence] = []
    for index, path in enumerate(paths, start=1):
        matches = [record for record in read_fasta(path) if record.id == header]
        if len(matches) != 1:
            raise ValidationError(
                f"Expected exactly one record with header {header!r} in {path}, "
                f"found {len(matches)}"
            )
        noisy_sequences.append(
            NoisySequence(
                path=path,
                internal_id=f"noisy_{index:06d}",
                sequence=matches[0].sequence,
            )
        )
    return noisy_sequences


def build_embedding_request(
    reference: ProteinSequence,
    noisy_sequences: list[NoisySequence],
    *,
    models: tuple[str, ...],
    device: str,
    precision: str,
):
    sequences = [
        {
            "id": "reference",
            "sequence": reference.sequence,
            "metadata": {"fasta_header": reference.id, "role": "reference"},
        }
    ]
    sequences.extend(
        {
            "id": item.internal_id,
            "sequence": item.sequence,
            "metadata": {
                "fasta_header": reference.id,
                "source_file": item.path.name,
                "role": "noisy",
            },
        }
        for item in noisy_sequences
    )
    return request_from_dict(
        {
            "models": list(models),
            "sequences": sequences,
            "representations": [
                {"method": "mean", "normalize": "l2"},
                {"method": "max", "normalize": "l2"},
                {"method": "residue", "normalize": "l2"},
            ],
            "execution": {
                "device": device,
                "precision": precision,
                "on_long_sequence": "error",
                "allow_partial": False,
            },
        },
        compare=False,
    )


def index_representations(output: dict[str, Any]) -> dict[tuple[str, str], list]:
    return {
        (item["method"], item["sequence_id"]): item["values"]
        for item in output["representations"]
    }


def sequence_scores(reference: str, target: str) -> dict[str, float]:
    reference = "".join(reference.split()).upper()
    target = "".join(target.split()).upper()
    return {
        f"sequence.{name}": SEQUENCE_METRICS[name](reference, target)
        for name in SEQUENCE_METRIC_NAMES
    }


def add_model_scores(
    row: dict[str, Any],
    *,
    model_id: str,
    representations: dict[tuple[str, str], list],
    target_id: str,
) -> None:
    for method in ("mean", "max"):
        reference_values = representations[(method, "reference")]
        target_values = representations[(method, target_id)]
        for metric_name in VECTOR_METRIC_NAMES:
            row[f"{model_id}.{method}.{metric_name}"] = VECTOR_METRICS[metric_name](
                reference_values, target_values
            )

    similarity = residue_similarity(
        representations[("residue", "reference")],
        representations[("residue", target_id)],
    )
    global_alignment = align_similarity(similarity, local=False)
    local_alignment = align_similarity(similarity, local=True)
    optimal_transport = sinkhorn_similarity(similarity)
    prefix = f"{model_id}.residue"
    row[f"{prefix}.global_alignment"] = global_alignment["normalized_score"]
    row[f"{prefix}.local_alignment"] = local_alignment["normalized_score"]
    row[f"{prefix}.best_match"] = symmetric_best_match(similarity)
    row[f"{prefix}.optimal_transport"] = optimal_transport["similarity"]


def collect_rows(
    reference: ProteinSequence,
    noisy_sequences: list[NoisySequence],
    embedded: dict[str, Any],
    noisy_directory: Path,
) -> list[dict[str, Any]]:
    model_representations = {
        model_id: index_representations(output)
        for model_id, output in embedded["models"].items()
    }
    rows: list[dict[str, Any]] = []
    reference_sequence = "".join(reference.sequence.split()).upper()
    for item in noisy_sequences:
        try:
            source_file = str(item.path.relative_to(noisy_directory))
        except ValueError:
            source_file = str(item.path)
        noisy_sequence = "".join(item.sequence.split()).upper()
        row: dict[str, Any] = {
            "noisy_file": source_file,
            "fasta_header": reference.id,
            "reference_length": len(reference_sequence),
            "noisy_length": len(noisy_sequence),
        }
        row.update(sequence_scores(reference_sequence, noisy_sequence))
        for model_id, representations in model_representations.items():
            add_model_scores(
                row,
                model_id=model_id,
                representations=representations,
                target_id=item.internal_id,
            )
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except OSError as exc:
        raise ValidationError(f"Cannot write output CSV {output_path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_fasta", type=Path, help="Reference protein FASTA file")
    parser.add_argument(
        "noisy_fasta_dir",
        type=Path,
        help="Directory containing noisy FASTA files (non-recursive)",
    )
    parser.add_argument(
        "--header",
        help="Full FASTA header to select when the reference contains multiple records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("noisy_sequence_scores.csv"),
        help="Single output CSV path (default: noisy_sequence_scores.csv)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        metavar="MODEL",
        help="Framework model IDs (default: esm2-8m esmc-300m)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Embedding device (default: auto)",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "bf16"),
        default="auto",
        help="Embedding precision (default: auto)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reference_path = args.reference_fasta.resolve()
        if not reference_path.is_file():
            raise ValidationError(f"Reference FASTA file does not exist: {args.reference_fasta}")
        noisy_directory = args.noisy_fasta_dir.resolve()
        reference = select_reference(reference_path, args.header)
        noisy_paths = discover_noisy_fastas(noisy_directory, reference_path)
        noisy_sequences = load_noisy_sequences(noisy_paths, reference.id)

        print(
            f"Matched header {reference.id!r} in {len(noisy_sequences)} noisy FASTA files.",
            file=sys.stderr,
        )
        print(
            f"Embedding {len(noisy_sequences) + 1} sequences with: "
            f"{', '.join(args.models)}",
            file=sys.stderr,
        )
        request = build_embedding_request(
            reference,
            noisy_sequences,
            models=tuple(args.models),
            device=args.device,
            precision=args.precision,
        )
        embedded = Pipeline().embed(request)
        rows = collect_rows(reference, noisy_sequences, embedded, noisy_directory)
        write_csv(rows, args.output)
    except FrameworkError as exc:
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} reference-to-noisy comparisons to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
