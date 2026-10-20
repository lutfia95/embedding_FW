"""Checks for batching, pair threads, streaming output, and numerical equivalence without PLM downloads."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from noisy_seq_analyser import (
    NoisySequence,
    build_embedding_request,
    collect_rows,
    embedding_batches,
    model_cache,
    run_batched_analysis,
    validate_model_cache,
)
from plms_framework import Pipeline
from plms_framework.adapters.deterministic import DeterministicAdapter
from plms_framework.errors import ValidationError
from plms_framework.models import ProteinSequence
from plms_framework.registry import ModelRegistry, default_registry


class ParallelScoringTests(unittest.TestCase):
    def test_model_wise_saves_each_model_before_next_and_clears_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            output = root / "scores"
            references = (ProteinSequence("protein", "ACD"),)
            noisy = [NoisySequence(root / "a.fa", "noisy_1", "protein", "ACE")]
            registry = ModelRegistry()
            manifest = default_registry.manifest("reference-deterministic-v1")

            def factory(model_manifest):
                if model_manifest.id == "second":
                    self.assertTrue((output / "first.csv").is_file())
                    self.assertEqual(list(cache.iterdir()), [])
                (cache / "checkpoint.bin").write_bytes(b"fake checkpoint")
                self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache.resolve() / "hub"))
                return DeterministicAdapter(model_manifest)

            for name in ("first", "second"):
                registry.register(replace(manifest, id=name, aliases=()), factory)
            request = build_embedding_request(
                references,
                noisy,
                models=("first", "second"),
                device="cpu",
                precision="auto",
            )
            with patch("plms_framework.pipeline.default_registry", registry):
                run_batched_analysis(
                    request,
                    references,
                    noisy,
                    root,
                    workers=1,
                    batch_size=2,
                    max_batch_tokens=20,
                    torch_threads=1,
                    cache_dir=cache,
                    model_wise=True,
                    output_dir=output,
                )
            self.assertEqual(list(cache.iterdir()), [])
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["first.csv", "second.csv"])
            for name, other in (("first", "second"), ("second", "first")):
                with (output / f"{name}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertIn("sequence.sequence_identity", rows[0])
                self.assertIn(f"{name}.mean.cosine_similarity", rows[0])
                self.assertFalse(any(key.startswith(f"{other}.") for key in rows[0]))

    def test_disposable_cache_cleanup_on_failure_preserves_symlink_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            target = root / "keep.txt"
            target.write_text("keep")
            before = os.environ.get("HF_HOME")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with model_cache(cache, disposable=True):
                    (cache / "hub").mkdir()
                    (cache / "hub" / "weights").write_text("checkpoint")
                    (cache / "link").symlink_to(target)
                    raise RuntimeError("failed")
            self.assertEqual(list(cache.iterdir()), [])
            self.assertEqual(target.read_text(), "keep")
            self.assertEqual(os.environ.get("HF_HOME"), before)

    def test_reject_cache_containing_input_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cache in (root, root / "noisy", root / "output"):
                with self.assertRaises(ValidationError):
                    validate_model_cache(cache, root / "ref.fa", root / "noisy", root / "output")

    def test_cli_writes_one_csv_with_multiple_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noisy = root / "noisy"
            noisy.mkdir()
            reference = root / "reference.fa"
            reference.write_text(">first\nMKTAY\n>second\nACD\n")
            (noisy / "one.fa").write_text(">second\nACD\n>first\nMKTAY\n")
            (noisy / "two.fa").write_text(">first\nMKTAA\n>second\nACDE\n")
            repository = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(repository / "noisy_seq_analyser.py"),
                    str(reference),
                    str(noisy),
                    "--output",
                    str(root / "scores.csv"),
                    "--models",
                    "reference-deterministic-v1",
                    "--workers",
                    "2",
                    "--batch-size",
                    "1",
                    "--cache-dir",
                    str(root / "scratch"),
                ],
                env={**os.environ, "PYTHONPATH": str(repository / "src")},
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["fasta_header"], "second")
            self.assertEqual(float(rows[0]["sequence.sequence_identity"]), 1.0)
            self.assertIn("2 pair worker threads", result.stderr)
            self.assertEqual(list((root / "scratch").iterdir()), [])
            model_output = root / "model_scores"
            command = result.args.copy()
            command[command.index("--output") + 1] = str(model_output)
            command.append("--model_wise")
            result = subprocess.run(
                command,
                env={**os.environ, "PYTHONPATH": str(repository / "src")},
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (model_output / "reference-deterministic-v1.csv").open(newline="") as handle:
                model_rows = list(csv.DictReader(handle))
                self.assertEqual(sorted(model_rows, key=lambda row: int(row["pair_index"])), rows)
            self.assertEqual(list((root / "scratch").iterdir()), [])

    def test_batches_obey_size_and_padding_budget(self):
        sequences = tuple(
            ProteinSequence(str(i), "A" * length) for i, length in enumerate((19, 2, 3, 4, 1))
        )
        batches = list(embedding_batches(sequences, 3, 12))
        self.assertEqual(sum(map(len, batches)), len(sequences))
        self.assertEqual(len({s.id for batch in batches for s in batch}), len(sequences))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * max(len(s.sequence) + 2 for s in batch), 12)

    def test_in_memory_serial_and_thread_workers_match_original_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references = (
                ProteinSequence("long reference", "MKTAYIAK"),
                ProteinSequence("short reference", "ACD"),
            )
            noisy = [
                NoisySequence(root / "a.fa", "noisy_1", "long reference", "MKTAYIAK"),
                NoisySequence(root / "b.fa", "noisy_2", "short reference", "ACDE"),
                NoisySequence(root / "b.fa", "noisy_3", "long reference", "MKTAYIAQ"),
            ]
            request = build_embedding_request(
                references,
                noisy,
                models=("reference-deterministic-v1",),
                device="cpu",
                precision="auto",
            )
            expected = collect_rows(references, noisy, Pipeline().embed(request), root)
            original_embed = DeterministicAdapter.embed
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    instances = []

                    def record_embed(adapter, *args, **kwargs):
                        instances.append(adapter)
                        return original_embed(adapter, *args, **kwargs)

                    with patch.object(DeterministicAdapter, "embed", record_embed):
                        actual = run_batched_analysis(
                            request,
                            references,
                            noisy,
                            root,
                            workers=workers,
                            batch_size=1,
                            max_batch_tokens=12,
                            torch_threads=1,
                            cache_dir=root,
                        )
                    self.assertEqual(len(instances), 5)
                    self.assertTrue(all(adapter is instances[0] for adapter in instances))
                    self.assertEqual(actual, expected)
                    self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
