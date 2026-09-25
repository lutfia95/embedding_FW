#!/usr/bin/env python3
"""Score matching noisy FASTA sequences against reference FASTA records.

Records are matched by their full FASTA header.  By default, every reference
record is matched against the records with the same header in each FASTA file
in ``NOISY_FASTA_DIR``. Embedding runs in bounded batches; CPU worker processes
score matched pairs using a temporary disk cache. Scores go to one CSV file.

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
import logging
import mmap
import multiprocessing
import os
import sys
import tempfile
import time
from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
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
from plms_framework.registry import ModelRegistry
from plms_framework.sequences import parse_fasta
from plms_framework.serialization import request_from_dict


DEFAULT_MODELS = ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b", "esm1b-650m", "esm1v-650m-1", "esm1v-650m-2", "esm1v-650m-3", "esm1v-650m-4", "esm1v-650m-5")
LOGGER = logging.getLogger("noisy_seq_analyser")
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
    """A matched noisy FASTA record and its safe framework identifier."""

    path: Path
    internal_id: str
    header: str
    sequence: str


def read_fasta(path: Path) -> tuple[ProteinSequence, ...]:
    """Read one FASTA file and add its path to parsing errors."""
    LOGGER.debug("Reading FASTA: %s", path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read FASTA file {path}: {exc}") from exc
    try:
        return parse_fasta(text)
    except FrameworkError as exc:
        raise ValidationError(f"Invalid FASTA file {path}: {exc}") from exc


def select_references(path: Path, requested_header: str | None) -> tuple[ProteinSequence, ...]:
    records = read_fasta(path)
    headers = [record.id for record in records]
    duplicates = sorted(header for header, count in Counter(headers).items() if count > 1)
    if duplicates:
        raise ValidationError(
            f"Reference FASTA {path} contains duplicate headers: {', '.join(duplicates)}"
        )
    if requested_header is None:
        LOGGER.info("Loaded %d reference records from %s", len(records), path)
        return records

    matches = tuple(record for record in records if record.id == requested_header)
    if len(matches) != 1:
        raise ValidationError(
            f"Expected exactly one reference record with header {requested_header!r} "
            f"in {path}, found {len(matches)}"
        )
    LOGGER.info("Selected reference header %r from %s", requested_header, path)
    return matches


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


def load_noisy_sequences(
    paths: list[Path], reference_headers: set[str]
) -> list[NoisySequence]:
    noisy_sequences: list[NoisySequence] = []
    LOGGER.info(
        "Scanning %d noisy FASTA files for %d reference headers",
        len(paths),
        len(reference_headers),
    )
    for file_index, path in enumerate(paths, start=1):
        LOGGER.info("[%d/%d] Scanning %s", file_index, len(paths), path.name)
        records = read_fasta(path)
        headers = [record.id for record in records]
        duplicates = sorted(
            {
                header
                for header, count in Counter(headers).items()
                if header in reference_headers and count > 1
            }
        )
        if duplicates:
            LOGGER.error("Duplicate reference headers found in %s: %s", path, ", ".join(duplicates))
            raise ValidationError(
                f"Noisy FASTA {path} contains duplicate matching headers: "
                f"{', '.join(duplicates)}"
            )
        matched_records = [record for record in records if record.id in reference_headers]
        matched_headers = {record.id for record in matched_records}
        missing_headers = reference_headers - matched_headers
        LOGGER.info(
            "[%d/%d] %s: %d records, %d matched, %d reference headers missing",
            file_index,
            len(paths),
            path.name,
            len(records),
            len(matched_records),
            len(missing_headers),
        )
        if missing_headers:
            LOGGER.debug(
                "Missing headers in %s (up to 10 shown): %s",
                path.name,
                ", ".join(sorted(missing_headers)[:10]),
            )
        for record in matched_records:
            if record.id in reference_headers:
                noisy_sequences.append(
                    NoisySequence(
                        path=path,
                        internal_id=f"noisy_{len(noisy_sequences) + 1:09d}",
                        header=record.id,
                        sequence=record.sequence,
                    )
                )
    if not noisy_sequences:
        LOGGER.error("No reference headers were found in any noisy FASTA file")
        raise ValidationError("No headers from the reference FASTA were found in the noisy files")
    LOGGER.info("Found %d matched noisy records across %d files", len(noisy_sequences), len(paths))
    return noisy_sequences


def build_embedding_request(
    references: tuple[ProteinSequence, ...],
    noisy_sequences: list[NoisySequence],
    *,
    models: tuple[str, ...],
    device: str,
    precision: str,
):
    reference_ids = {
        reference.id: f"reference_{index:06d}"
        for index, reference in enumerate(references, start=1)
    }
    sequences = [
        {
            "id": reference_ids[reference.id],
            "sequence": reference.sequence,
            "metadata": {"fasta_header": reference.id, "role": "reference"},
        }
        for reference in references
    ]
    sequences.extend(
        {
            "id": item.internal_id,
            "sequence": item.sequence,
            "metadata": {
                "fasta_header": item.header,
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
    reference_id: str,
    target_id: str,
) -> None:
    for method in ("mean", "max"):
        reference_values = representations[(method, reference_id)]
        target_values = representations[(method, target_id)]
        for metric_name in VECTOR_METRIC_NAMES:
            row[f"{model_id}.{method}.{metric_name}"] = VECTOR_METRICS[metric_name](
                reference_values, target_values
            )

    similarity = residue_similarity(
        representations[("residue", reference_id)],
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
    references: tuple[ProteinSequence, ...],
    noisy_sequences: list[NoisySequence],
    embedded: dict[str, Any],
    noisy_directory: Path,
) -> list[dict[str, Any]]:
    model_representations = {
        model_id: index_representations(output)
        for model_id, output in embedded["models"].items()
    }
    rows: list[dict[str, Any]] = []
    reference_by_header = {reference.id: reference for reference in references}
    reference_ids = {
        reference.id: f"reference_{index:06d}"
        for index, reference in enumerate(references, start=1)
    }
    for row_index, item in enumerate(noisy_sequences, start=1):
        reference = reference_by_header[item.header]
        reference_sequence = "".join(reference.sequence.split()).upper()
        try:
            source_file = str(item.path.relative_to(noisy_directory))
        except ValueError:
            source_file = str(item.path)
        noisy_sequence = "".join(item.sequence.split()).upper()
        row: dict[str, Any] = {
            "noisy_file": source_file,
            "fasta_header": item.header,
            "reference_length": len(reference_sequence),
            "noisy_length": len(noisy_sequence),
        }
        row.update(sequence_scores(reference_sequence, noisy_sequence))
        for model_id, representations in model_representations.items():
            add_model_scores(
                row,
                model_id=model_id,
                representations=representations,
                reference_id=reference_ids[item.header],
                target_id=item.internal_id,
            )
        rows.append(row)
        LOGGER.debug(
            "Scored header %r from %s (%d/%d)",
            item.header,
            item.path.name,
            row_index,
            len(noisy_sequences),
        )
        if row_index % 1000 == 0 or row_index == len(noisy_sequences):
            LOGGER.info("Scored %d/%d reference-to-noisy pairs", row_index, len(noisy_sequences))
    return rows


def available_cpus() -> int:
    """Respect CPU affinity where available (e.g. a scheduler allocation)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return (getattr(os, "process_cpu_count", os.cpu_count)() or 1)


def embedding_batches(sequences, batch_size: int, max_batch_tokens: int):
    """Length bucketing with a cap on padded tokens, including boundary tokens."""
    ordered = sorted(sequences, key=lambda item: len("".join(item.sequence.split())))
    batch = []
    for sequence in ordered:
        length = len("".join(sequence.sequence.split())) + 2
        if batch and (
            len(batch) >= batch_size or (len(batch) + 1) * length > max_batch_tokens
        ):
            yield tuple(batch)
            batch = []
        batch.append(sequence)
    if batch:
        yield tuple(batch)


def cache_model_embeddings(pipeline, request, cache_path, batch_size, max_batch_tokens):
    """Write exact float64 values once; workers map only the requested pair."""
    batches = list(embedding_batches(request.sequences, batch_size, max_batch_tokens))
    index = {}
    started = time.perf_counter()
    completed = 0
    with cache_path.open("wb") as handle:
        for number, sequences in enumerate(batches, start=1):
            LOGGER.info(
                "%s embedding batch %d/%d: %d sequences, longest=%d residues",
                request.models[0], number, len(batches), len(sequences),
                max(len("".join(item.sequence.split())) for item in sequences),
            )
            embedded = pipeline.embed(replace(request, sequences=sequences))
            model_id, output = next(iter(embedded["models"].items()))
            for item in output["representations"]:
                values = item["values"]
                flattened = (
                    (value for row in values for value in row)
                    if item["method"] == "residue" else iter(values)
                )
                data = array("d", flattened)
                descriptor = (handle.tell(), len(data), tuple(item["shape"]))
                data.tofile(handle)
                index.setdefault(item["sequence_id"], {})[item["method"]] = descriptor
            completed += len(sequences)
            LOGGER.info(
                "%s embedded %d/%d sequences in %.1fs; cache %.1f MiB",
                model_id, completed, len(request.sequences), time.perf_counter() - started,
                handle.tell() / 1024**2,
            )
            del embedded, output
    return model_id, index


def _score_mapped_pair(mapped, model_id, reference_spec, target_spec):
    representations = {}
    for sequence_id, spec in (("reference", reference_spec), ("target", target_spec)):
        for method, (offset, size, shape) in spec.items():
            view = memoryview(mapped)[offset:offset + size * array("d").itemsize].cast("d")
            representations[(method, sequence_id)] = (
                [view[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]
                if method == "residue" else view
            )
    scores = {}
    try:
        add_model_scores(
            scores, model_id=model_id, representations=representations,
            reference_id="reference", target_id="target",
        )
        return scores
    finally:
        # Release exported buffers before the caller closes the memory map,
        # including when scoring raises an exception.
        for values in representations.values():
            if isinstance(values, list):
                for row in values:
                    row.release()
            else:
                values.release()
        view.release()


def score_cached_pair(task):
    """Spawn-safe worker: receives paths and offsets, never a model or full embeddings."""
    row_index, cache_path, model_id, ref_spec, target_spec, baseline_sequences = task
    try:
        with open(cache_path, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                scores = _score_mapped_pair(mapped, model_id, ref_spec, target_spec)
        if baseline_sequences is not None:
            scores.update(sequence_scores(*baseline_sequences))
        return row_index, scores
    except Exception as exc:
        raise RuntimeError(f"Scoring pair {row_index + 1} for {model_id} failed: {exc}") from exc


def score_tasks(tasks, rows, workers, model_id):
    started = last_report = time.perf_counter()
    completed = 0
    total = len(rows)

    def report():
        elapsed = time.perf_counter() - started
        rate = completed / max(elapsed, 1e-9)
        LOGGER.info(
            "%s scoring: %d/%d pairs, %.2f pairs/s, elapsed %.1fs, ETA %s",
            model_id, completed, total, rate, elapsed,
            f"{(total - completed) / rate:.1f}s" if rate else "pending first result",
        )

    if workers == 1:
        for task in tasks:
            row_index, scores = score_cached_pair(task)
            rows[row_index].update(scores)
            completed += 1
            if time.perf_counter() - last_report >= 10 or completed == total:
                report()
                last_report = time.perf_counter()
        return

    # Spawn avoids inheriting initialized torch/CUDA state. Bound the queue so
    # a large proteome does not create millions of pending futures.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        iterator = iter(tasks)
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                try:
                    pending.add(pool.submit(score_cached_pair, next(iterator)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                row_index, scores = future.result()
                rows[row_index].update(scores)
                completed += 1
                LOGGER.debug("%s finished pair %d/%d", model_id, row_index + 1, total)
            if time.perf_counter() - last_report >= 10 or completed == total:
                report()
                last_report = time.perf_counter()


def run_batched_analysis(request, references, noisy_sequences, noisy_directory, *,
                         workers, batch_size, max_batch_tokens, torch_threads, cache_dir=None):
    reference_ids = {
        reference.id: f"reference_{number:06d}"
        for number, reference in enumerate(references, start=1)
    }
    reference_by_header = {item.id: item.sequence for item in references}
    rows = [
        {
            "noisy_file": str(item.path.relative_to(noisy_directory)),
            "fasta_header": item.header,
            "reference_length": len("".join(reference_by_header[item.header].split())),
            "noisy_length": len("".join(item.sequence.split())),
        }
        for item in noisy_sequences
    ]
    pipeline = Pipeline()
    # The deterministic adapter needs no torch installation.
    if any(pipeline.registry.manifest(model).backend != "built_in"
           for model in request.models):
        try:
            import torch
        except ImportError:
            pass  # The adapter provides the actionable dependency error.
        else:
            torch.set_num_threads(torch_threads)
    workers = min(workers, len(rows))
    LOGGER.info(
        "Execution: %d scoring workers; embedding batch size=%d, token budget=%d, CPU threads=%d",
        workers, batch_size, max_batch_tokens, torch_threads,
    )
    for number, model in enumerate(request.models, start=1):
        LOGGER.info("Model %d/%d: %s", number, len(request.models), model)
        with tempfile.TemporaryDirectory(prefix="noisy-embeddings-", dir=cache_dir) as scratch:
            cache_path = Path(scratch) / "embeddings.bin"
            LOGGER.info("Temporary embedding cache: %s", cache_path)
            # The general registry constructs fresh adapters on each lookup.
            # Reuse this instance so batches do not reload the checkpoint.
            adapter = pipeline.registry.adapter(model)
            batch_registry = ModelRegistry()
            batch_registry.register(adapter.manifest, lambda manifest: adapter)
            batch_pipeline = Pipeline(batch_registry)
            try:
                model_id, index = cache_model_embeddings(
                    batch_pipeline, replace(request, models=(model,)), cache_path,
                    batch_size, max_batch_tokens,
                )
            finally:
                adapter.unload()
            tasks = (
                (
                    i, str(cache_path), model_id, index[reference_ids[item.header]],
                    index[item.internal_id],
                    (reference_by_header[item.header], item.sequence) if number == 1 else None,
                )
                for i, item in enumerate(noisy_sequences)
            )
            LOGGER.info("%s: scoring %d pairs with %d workers", model_id, len(rows), workers)
            score_tasks(tasks, rows, workers, model_id)
        LOGGER.info("%s complete; temporary embedding cache removed", model_id)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    LOGGER.info("Writing %d score rows to %s", len(rows), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
        LOGGER.info("CSV written successfully: %s (%d columns)", output_path, len(rows[0]))
    except OSError as exc:
        LOGGER.exception("Failed writing score CSV %s", output_path)
        raise ValidationError(f"Cannot write output CSV {output_path}: {exc}") from exc


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


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
        nargs="?",
        const="",
        help="Optionally select one full reference header; omit or leave bare to process all",
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
        help="Framework model IDs (default: %(default)s)",
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
    parser.add_argument(
        "--workers", type=positive_int,
        default=min(8, max(1, available_cpus() - 1)),
        help="Scoring processes (default: available CPUs minus one, capped at 8; minimum 1)",
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=4,
        help="Maximum sequences per embedding batch (default: 4)",
    )
    parser.add_argument(
        "--max-batch-tokens", type=positive_int, default=2048,
        help="Padded token budget per batch (default: 2048); longer sequences run alone",
    )
    parser.add_argument(
        "--torch-threads", type=positive_int, default=min(8, available_cpus()),
        help="CPU threads during embedding (default: available CPUs capped at 8)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Parent for temporary embedding caches (default: system temporary directory)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO; use DEBUG for missing-header examples)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    started = time.perf_counter()
    LOGGER.info("Starting noisy sequence analysis")
    LOGGER.info("Reference FASTA: %s", args.reference_fasta)
    LOGGER.info("Noisy FASTA directory: %s", args.noisy_fasta_dir)
    LOGGER.info("Output CSV: %s", args.output)
    LOGGER.info("Models: %s", ", ".join(args.models))
    try:
        reference_path = args.reference_fasta.resolve()
        if not reference_path.is_file():
            raise ValidationError(f"Reference FASTA file does not exist: {args.reference_fasta}")
        noisy_directory = args.noisy_fasta_dir.resolve()
        requested_header = args.header or None
        references = select_references(reference_path, requested_header)
        noisy_paths = discover_noisy_fastas(noisy_directory, reference_path)
        noisy_sequences = load_noisy_sequences(
            noisy_paths, {reference.id for reference in references}
        )

        LOGGER.info(
            "Input matching complete: %d reference record(s), %d matched noisy record(s)",
            len(references),
            len(noisy_sequences),
        )
        LOGGER.info(
            "Starting embedding for %d total sequences using %s",
            len(references) + len(noisy_sequences),
            ", ".join(args.models),
        )
        request = build_embedding_request(
            references,
            noisy_sequences,
            models=tuple(args.models),
            device=args.device,
            precision=args.precision,
        )
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
        rows = run_batched_analysis(
            request, references, noisy_sequences, noisy_directory,
            workers=args.workers, batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens, torch_threads=args.torch_threads,
            cache_dir=args.cache_dir,
        )
        write_csv(rows, args.output)
    except FrameworkError as exc:
        LOGGER.error("Analysis stopped: %s", exc)
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        LOGGER.exception("Analysis failed: %s", exc)
        return 1

    LOGGER.info(
        "Analysis complete: wrote %d comparison rows to %s in %.2f seconds",
        len(rows),
        args.output.resolve(),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
