#!/usr/bin/env python3
"""Score matching noisy FASTA sequences against reference FASTA records.

Records are matched by their full FASTA header.  By default, every reference
record is matched against the records with the same header in each FASTA file
in ``NOISY_FASTA_DIR``. Worker threads embed and score complete pairs in memory.
Each completed pair is flushed to a model CSV immediately; no embedding arrays
are written to disk.
With --model_wise, --output is a directory containing one CSV per model.

Example:

    python3.12 noisy_seq_analyser.py reference.fasta noisy_fastas \
        --output noisy_sequence_scores.csv

    python3.12 noisy_seq_analyser.py reference.fasta noisy_fastas \
        --model_wise --output model_scores --cache-dir model_cache

In model-wise mode, each CSV contains reference-to-noisy scores from just one
model, plus sequence metrics. The entire supplied cache directory is emptied
after every model (including on failure), so use a dedicated disposable folder.
Checkpoint downloads are directed there; existing global caches are unchanged.

The default model and metric choices mirror the pairwise scores demonstrated by
``python_ex.py``.  The first run may download the ESM-2 and ESM-C checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, local
from typing import Any
from urllib.parse import quote

from plms_framework import Pipeline
from plms_framework.errors import FrameworkError, ResourceLimitError, ValidationError
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


DEFAULT_MODELS = ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b", "esm1b-650m", "esm1v-650m-1", "esm1v-650m-2", "esm1v-650m-3", "esm1v-650m-4", "esm1v-650m-5", "esmc-300m", "esmc-600m", "esmc-6b")
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


def load_noisy_sequences(paths: list[Path], reference_headers: set[str]) -> list[NoisySequence]:
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
                f"Noisy FASTA {path} contains duplicate matching headers: {', '.join(duplicates)}"
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
        (item["method"], item["sequence_id"]): item["values"] for item in output["representations"]
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
        model_id: index_representations(output) for model_id, output in embedded["models"].items()
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
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def embedding_batches(sequences, batch_size: int, max_batch_tokens: int):
    """Length bucketing with a cap on padded tokens, including boundary tokens."""
    ordered = sorted(sequences, key=lambda item: len("".join(item.sequence.split())))
    batch = []
    for sequence in ordered:
        length = len("".join(sequence.sequence.split())) + 2
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * length > max_batch_tokens):
            yield tuple(batch)
            batch = []
        batch.append(sequence)
    if batch:
        yield tuple(batch)


def model_score_fields(model_id):
    return [
        *(
            f"{model_id}.{method}.{metric}"
            for method in ("mean", "max")
            for metric in VECTOR_METRIC_NAMES
        ),
        *(
            f"{model_id}.residue.{metric}"
            for metric in ("global_alignment", "local_alignment", "best_match", "optimal_transport")
        ),
    ]


class ScoreOutput:
    """Flush model CSVs immediately; optionally merge only scalar scores on disk."""

    def __init__(self, path, model_ids, model_wise):
        self.path = Path(path)
        self.model_ids = model_ids
        self.model_wise = model_wise
        self.directory = (
            self.path if model_wise else self.path.with_name(self.path.stem + ".models")
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scratch = None
        self.database = None
        self.handle = None
        self.metadata = ["noisy_file", "fasta_header", "reference_length", "noisy_length"]
        self.baselines = [f"sequence.{name}" for name in SEQUENCE_METRIC_NAMES]
        if not model_wise:
            # Scores only, never embeddings. This bounds memory during wide CSV merging.
            self.scratch = tempfile.TemporaryDirectory(prefix=".score-merge-", dir=self.path.parent)
            self.database = sqlite3.connect(Path(self.scratch.name) / "scores.sqlite")
            self.database.execute(
                "CREATE TABLE scores (pair_index INTEGER, model_index INTEGER, data TEXT, "
                "PRIMARY KEY (pair_index, model_index))"
            )

    def start_model(self, model_id):
        self.finish_model()
        path = self.directory / f"{quote(model_id, safe='')}.csv"
        self.handle = path.open("w", encoding="utf-8", newline="")
        fields = [
            "pair_index",
            *self.metadata,
            *self.baselines,
            *model_score_fields(model_id),
            f"{model_id}.status",
            f"{model_id}.error",
        ]
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()
        self.handle.flush()
        LOGGER.info("Streaming pair scores to %s", path)

    def write(self, pair_index, model_index, row):
        model_id = self.model_ids[model_index]
        recorded = {**row, f"{model_id}.status": row.get(f"{model_id}.status", "ok")}
        self.writer.writerow({"pair_index": pair_index, **recorded})
        self.handle.flush()
        if self.database is not None:
            self.database.execute(
                "INSERT INTO scores VALUES (?, ?, ?)",
                (pair_index, model_index, json.dumps(recorded)),
            )

    def finish_model(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.database is not None:
            self.database.commit()

    def finalize(self):
        self.finish_model()
        if self.database is None:
            return
        fields = ["pair_index", *self.metadata, *self.baselines]
        for model_id in self.model_ids:
            fields.extend(
                [*model_score_fields(model_id), f"{model_id}.status", f"{model_id}.error"]
            )
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            previous = None
            merged = {}
            for pair_index, data in self.database.execute(
                "SELECT pair_index, data FROM scores ORDER BY pair_index, model_index"
            ):
                if previous is not None and pair_index != previous:
                    writer.writerow(merged)
                    merged = {}
                merged.update(json.loads(data))
                merged["pair_index"] = pair_index
                previous = pair_index
            if previous is not None:
                writer.writerow(merged)
        temporary.replace(self.path)

    def close(self):
        self.finish_model()
        if self.database is not None:
            self.database.close()
            self.database = None
        if self.scratch is not None:
            self.scratch.cleanup()


def pair_row(item, reference, noisy_directory):
    try:
        source = str(item.path.relative_to(noisy_directory))
    except ValueError:
        source = str(item.path)
    return {
        "noisy_file": source,
        "fasta_header": item.header,
        "reference_length": len("".join(reference.sequence.split())),
        "noisy_length": len("".join(item.sequence.split())),
    }


def cpu_worker_adapter(adapter):
    """Separate tokenizer/module caches per CPU thread while sharing model weights."""
    model = getattr(adapter, "_model", None)
    if model is None:
        return adapter  # The built-in deterministic adapter has no mutable model state.
    shared_tensors = {id(tensor): tensor for tensor in (*model.parameters(), *model.buffers())}
    return copy.deepcopy(adapter, shared_tensors)


def score_in_memory_pair(
    item,
    reference,
    request,
    pipeline,
    noisy_directory,
    *,
    batch_size,
    max_batch_tokens,
    inference_lock,
    long_sequences,
):
    """Only scalar scores escape this worker; pair embeddings are local temporaries."""
    model_id = request.models[0]
    row = pair_row(item, reference, noisy_directory)
    manifest = pipeline.registry.manifest(model_id)
    limit = manifest.max_native_length
    if limit is not None and max(row["reference_length"], row["noisy_length"]) > limit:
        error = ResourceLimitError(
            f"Sequences exceed native length {limit}; truncation is forbidden",
            context={"model": model_id, "header": item.header, "file": str(item.path)},
        )
        if long_sequences == "error":
            raise error
        row.update({f"{model_id}.status": "skipped_long_sequence", f"{model_id}.error": str(error)})
        LOGGER.warning(
            "Skipping %s pair %r in %s: %s", model_id, item.header, item.path.name, error
        )
        return row
    target = ProteinSequence(item.internal_id, item.sequence)
    same_sequence = (
        "".join(reference.sequence.split()).upper() == "".join(item.sequence.split()).upper()
    )
    sequences = (reference,) if same_sequence else (reference, target)
    representations = {}
    try:
        for batch in embedding_batches(sequences, batch_size, max_batch_tokens):
            with inference_lock if inference_lock is not None else nullcontext():
                embedded = pipeline.embed(replace(request, sequences=batch))
            output = embedded["models"][model_id]
            representations.update(index_representations(output))
            del embedded, output
        if same_sequence:
            for method in ("mean", "max", "residue"):
                representations[(method, target.id)] = representations[(method, reference.id)]
        add_model_scores(
            row,
            model_id=model_id,
            representations=representations,
            reference_id=reference.id,
            target_id=target.id,
        )
        row.update(sequence_scores(reference.sequence, item.sequence))
        return row
    finally:
        representations.clear()


def make_pair_worker(
    adapter,
    model_request,
    reference_by_header,
    noisy_directory,
    *,
    batch_size,
    max_batch_tokens,
    long_sequences,
):
    """Bind one model's state for its worker pool, independent of later models."""
    thread_state = local()
    clone_lock = Lock()
    worker_adapters = []
    accelerator = str(getattr(adapter, "_device", None) or "cpu") != "cpu"
    inference_lock = Lock() if accelerator else None

    def score(item):
        if not hasattr(thread_state, "pipeline"):
            with clone_lock:
                worker_adapter = adapter if accelerator else cpu_worker_adapter(adapter)
                if worker_adapter is not adapter:
                    worker_adapters.append(worker_adapter)
            worker_registry = ModelRegistry()
            worker_registry.register(worker_adapter.manifest, lambda manifest: worker_adapter)
            thread_state.pipeline = Pipeline(worker_registry)
        return score_in_memory_pair(
            item,
            reference_by_header[item.header],
            model_request,
            thread_state.pipeline,
            noisy_directory,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            inference_lock=inference_lock,
            long_sequences=long_sequences,
        )

    return score, worker_adapters


def run_batched_analysis(
    request,
    references,
    noisy_sequences,
    noisy_directory,
    *,
    workers,
    batch_size,
    max_batch_tokens,
    torch_threads,
    cache_dir=None,
    output_path=None,
    model_wise=False,
    output_dir=None,
    long_sequences="skip",
):
    """Send complete pairs to bounded threads, with no embedding cache on disk.

    CLI runs stream scores and return the pair count. Calls without output_path
    retain the legacy list-of-wide-score-rows return value for Python callers.
    cache_dir is accepted for compatibility and is never used for embeddings.
    """
    if min(workers, batch_size, max_batch_tokens, torch_threads) < 1:
        raise ValidationError(
            "workers, batch_size, max_batch_tokens and torch_threads must be positive"
        )
    if long_sequences not in {"skip", "error"}:
        raise ValidationError("long_sequences must be 'skip' or 'error'")
    if not noisy_sequences:
        raise ValidationError("No matched noisy pairs to score")
    if model_wise:
        output_path = output_path or output_dir
        if output_path is None:
            raise ValidationError("Model-wise analysis requires an output directory")
    registry = Pipeline().registry
    model_ids = tuple(dict.fromkeys(registry.resolve_id(model) for model in request.models))
    reference_by_header = {
        reference.id: replace(reference, id=f"reference_{number:06d}")
        for number, reference in enumerate(references, start=1)
    }
    rows = (
        [
            pair_row(item, reference_by_header[item.header], noisy_directory)
            for item in noisy_sequences
        ]
        if output_path is None
        else None
    )
    torch_module = None
    previous_threads = None
    if any(registry.manifest(model).backend != "built_in" for model in model_ids):
        try:
            import torch as torch_module
        except ImportError:
            pass
        else:
            previous_threads = torch_module.get_num_threads()
            torch_module.set_num_threads(torch_threads)
    workers = min(workers, len(noisy_sequences))
    LOGGER.info(
        "Execution: %d pair worker threads; batch size=%d, token budget=%d, "
        "torch threads=%d; embeddings stay in memory only",
        workers,
        batch_size,
        max_batch_tokens,
        torch_threads,
    )
    sink = None
    try:
        if output_path is not None:
            sink = ScoreOutput(output_path, model_ids, model_wise)
        for model_index, model_id in enumerate(model_ids):
            with model_cache(cache_dir, disposable=model_wise):
                adapter = registry.adapter(model_id)
                worker_adapters = []
                started = last_report = time.perf_counter()
                completed = 0
                try:
                    adapter.validate_execution(request.execution)
                    adapter.load(
                        device=request.execution.device, precision=request.execution.precision
                    )
                    model_request = replace(request, models=(model_id,))
                    if sink is not None:
                        sink.start_model(model_id)

                    score, worker_adapters = make_pair_worker(
                        adapter,
                        model_request,
                        reference_by_header,
                        noisy_directory,
                        batch_size=batch_size,
                        max_batch_tokens=max_batch_tokens,
                        long_sequences=long_sequences,
                    )

                    with ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="noisy-pair"
                    ) as pool:
                        iterator = iter(enumerate(noisy_sequences))
                        pending = {}
                        exhausted = False
                        try:
                            while pending or not exhausted:
                                while not exhausted and len(pending) < workers * 2:
                                    try:
                                        pair_index, item = next(iterator)
                                    except StopIteration:
                                        exhausted = True
                                    else:
                                        pending[pool.submit(score, item)] = pair_index
                                if not pending:
                                    break
                                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                                for future in done:
                                    pair_index = pending.pop(future)
                                    scores = future.result()
                                    if sink is not None:
                                        sink.write(pair_index, model_index, scores)
                                    else:
                                        rows[pair_index].update(scores)
                                    completed += 1
                                now = time.perf_counter()
                                if now - last_report >= 10 or completed == len(noisy_sequences):
                                    elapsed = now - started
                                    LOGGER.info(
                                        "%s: %d/%d pairs scored and written, %.2f pairs/s",
                                        model_id,
                                        completed,
                                        len(noisy_sequences),
                                        completed / max(elapsed, 1e-9),
                                    )
                                    last_report = now
                        finally:
                            for future in pending:
                                future.cancel()
                            pending.clear()
                    if sink is not None:
                        sink.finish_model()
                finally:
                    for worker_adapter in worker_adapters:
                        worker_adapter.unload()
                    worker_adapters.clear()
                    adapter.unload()
                    gc.collect()
                    if torch_module is not None and torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()
        if sink is not None:
            sink.finalize()
        return rows if rows is not None else len(noisy_sequences)
    finally:
        if sink is not None:
            sink.close()
        if previous_threads is not None:
            torch_module.set_num_threads(previous_threads)


@contextmanager
def model_cache(cache_dir, *, disposable=False):
    """Keep checkpoint downloads in a disposable cache per model.

    Set download locations before the lazy backend imports. Each model reuses
    the same paths because Hugging Face reads these variables at import time.
    """
    if not disposable:
        with tempfile.TemporaryDirectory(prefix="noisy-model-", dir=cache_dir) as scratch:
            yield scratch
        return
    if cache_dir is None:
        raise ValidationError("Model-wise analysis requires a cache directory")
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    locations = {
        "HF_HOME": str(cache_dir),
        "HF_HUB_CACHE": str(cache_dir / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache_dir / "hub"),
        "TRANSFORMERS_CACHE": str(cache_dir / "hub"),
        "HF_XET_CACHE": str(cache_dir / "xet"),
        "HF_MODULES_CACHE": str(cache_dir / "modules"),
        "TORCH_HOME": str(cache_dir / "torch"),
    }
    previous = {key: os.environ.get(key) for key in locations}
    os.environ.update(locations)
    try:
        yield str(cache_dir)
    finally:
        try:
            for path in cache_dir.iterdir():
                if path.is_symlink() or not path.is_dir():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            LOGGER.info("Emptied model cache: %s", cache_dir)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def validate_model_cache(cache_dir, reference_path, noisy_directory, output_dir):
    cache = cache_dir.resolve()
    protected = [
        reference_path.resolve(),
        noisy_directory.resolve(),
        output_dir.resolve(),
        Path(__file__).resolve(),
        Path.cwd().resolve(),
        Path.home().resolve(),
    ]
    if any(path == cache or cache in path.parents for path in protected):
        raise ValidationError(
            "--cache-dir must be a dedicated disposable directory, separate from "
            "inputs, output, source code, the working directory and home directory"
        )


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
        help="CSV path, or directory with --model_wise "
        "(defaults: noisy_sequence_scores.csv / noisy_sequence_scores)",
    )
    parser.add_argument(
        "--model_wise",
        "--model-wise",
        action="store_true",
        help="Score and save each model separately as <model-id>.csv in --output; "
        "remove its temporary cache before starting the next model",
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
        "--workers",
        type=positive_int,
        default=min(8, max(1, available_cpus() - 1)),
        help="Complete pair worker threads (default: available CPUs minus one, capped at 8; minimum 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
        help="Maximum sequences per embedding batch (default: 4)",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=positive_int,
        default=2048,
        help="Padded token budget per batch (default: 2048); longer sequences run alone",
    )
    parser.add_argument(
        "--torch-threads",
        type=positive_int,
        default=1,
        help="PyTorch threads per operation; pair concurrency uses --workers (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="With --model_wise, a disposable directory for checkpoint downloads "
        "whose entire contents are deleted after each model; no embeddings are saved "
        "(default: system temporary directory)",
    )
    parser.add_argument(
        "--long-sequences",
        choices=("skip", "error"),
        default="skip",
        help="Record over-limit pairs and continue, or stop; never truncate (default: skip)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO; use DEBUG for missing-header examples)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(
            "noisy_sequence_scores" if args.model_wise else "noisy_sequence_scores.csv"
        )
    return args


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
    LOGGER.info("Output %s: %s", "directory" if args.model_wise else "CSV", args.output)
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
            "Starting pair analysis for %d matched pairs using %s",
            len(noisy_sequences),
            ", ".join(args.models),
        )
        request = build_embedding_request(
            references,
            noisy_sequences,
            models=tuple(args.models),
            device=args.device,
            precision=args.precision,
        )
        with tempfile.TemporaryDirectory(prefix="noisy-model-cache-") as default_cache:
            if args.model_wise:
                args.cache_dir = args.cache_dir or Path(default_cache)
                validate_model_cache(
                    args.cache_dir,
                    reference_path,
                    noisy_directory,
                    args.output,
                )
                if args.output.exists() and not args.output.is_dir():
                    raise ValidationError("--output must be a directory with --model_wise")
            if args.cache_dir is not None:
                args.cache_dir.mkdir(parents=True, exist_ok=True)
            pair_count = run_batched_analysis(
                request,
                references,
                noisy_sequences,
                noisy_directory,
                workers=args.workers,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                torch_threads=args.torch_threads,
                cache_dir=args.cache_dir,
                model_wise=args.model_wise,
                output_path=args.output,
                long_sequences=args.long_sequences,
            )
    except FrameworkError as exc:
        LOGGER.error("Analysis stopped: %s", exc)
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except (OSError, RuntimeError):
        LOGGER.exception("Analysis failed")
        return 1

    LOGGER.info(
        "Analysis complete: wrote %d comparison rows to %s in %.2f seconds",
        pair_count * (len(request.models) if args.model_wise else 1),
        args.output.resolve(),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
