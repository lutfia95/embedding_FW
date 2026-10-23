from __future__ import annotations

import csv
from pathlib import Path

import pytest

from examples.noisy_seq_analyser import (
    build_embedding_request,
    collect_rows,
    discover_noisy_fastas,
    load_noisy_sequences,
    select_references,
    write_csv,
)
from plms_framework import Pipeline
from plms_framework.errors import ValidationError


def write_fasta(path: Path, header: str, sequence: str) -> None:
    path.write_text(f">{header}\n{sequence}\n", encoding="utf-8")


def test_deterministic_reference_to_noisy_scores(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.fasta"
    noisy_directory = tmp_path / "noisy"
    noisy_directory.mkdir()
    write_fasta(reference_path, "protein one", "MKTAYIAK")
    write_fasta(noisy_directory / "b.faa", "protein one", "MKTAYIAQ")
    write_fasta(noisy_directory / "a.fasta", "protein one", "MKTAYIAK")

    reference = select_references(reference_path, None)
    paths = discover_noisy_fastas(noisy_directory, reference_path)
    noisy = load_noisy_sequences(paths, {reference[0].id})
    request = build_embedding_request(
        reference,
        noisy,
        models=("reference-deterministic-v1",),
        device="cpu",
        precision="auto",
    )
    embedded = Pipeline().embed(request)
    rows = collect_rows(reference, noisy, embedded, noisy_directory)

    assert [row["noisy_file"] for row in rows] == ["a.fasta", "b.faa"]
    assert rows[0]["sequence.sequence_identity"] == 1.0
    assert rows[0]["reference-deterministic-v1.mean.cosine_similarity"] == pytest.approx(1.0)
    assert rows[0]["reference-deterministic-v1.residue.best_match"] == pytest.approx(1.0)
    assert rows[1]["sequence.sequence_identity"] < 1.0

    output_path = tmp_path / "scores.csv"
    write_csv(rows, output_path)
    with output_path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert written[0]["noisy_file"] == "a.fasta"


def test_missing_matching_header_is_rejected(tmp_path: Path) -> None:
    noisy_path = tmp_path / "noisy.fasta"
    write_fasta(noisy_path, "different", "MKTAYIAK")

    with pytest.raises(ValidationError, match="No headers from the reference"):
        load_noisy_sequences([noisy_path], {"wanted"})


def test_multirecord_reference_defaults_to_all_records(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.fasta"
    reference_path.write_text(">first\nAAAA\n>second\nCCCC\n", encoding="utf-8")

    assert [record.id for record in select_references(reference_path, None)] == [
        "first",
        "second",
    ]
    assert select_references(reference_path, "second")[0].sequence == "CCCC"


def test_multiple_headers_match_per_file(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.fasta"
    noisy_path = tmp_path / "noisy.fasta"
    reference_path.write_text(">first\nAAAA\n>second\nCCCC\n", encoding="utf-8")
    noisy_path.write_text(">first\nAAAT\n>second\nCCCT\n", encoding="utf-8")
    references = select_references(reference_path, None)
    noisy = load_noisy_sequences([noisy_path], {"first", "second"})

    request = build_embedding_request(
        references,
        noisy,
        models=("reference-deterministic-v1",),
        device="cpu",
        precision="auto",
    )
    rows = collect_rows(
        references,
        noisy,
        Pipeline().embed(request),
        tmp_path,
    )

    assert [row["fasta_header"] for row in rows] == ["first", "second"]
    assert [row["noisy_file"] for row in rows] == ["noisy.fasta", "noisy.fasta"]
    assert all(row["sequence.sequence_identity"] < 1.0 for row in rows)
