"""Checks for batching, process isolation, and numerical equivalence without PLM downloads."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from noisy_seq_analyser import (
    NoisySequence,
    build_embedding_request,
    collect_rows,
    embedding_batches,
    run_batched_analysis,
)
from plms_framework import Pipeline
from plms_framework.adapters.deterministic import DeterministicAdapter
from plms_framework.models import ProteinSequence


class ParallelScoringTests(unittest.TestCase):
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
                [sys.executable, "-B", str(repository / "noisy_seq_analyser.py"),
                 str(reference), str(noisy), "--output", str(root / "scores.csv"),
                 "--models", "reference-deterministic-v1", "--workers", "2",
                 "--batch-size", "1", "--cache-dir", str(root / "scratch")],
                env={**os.environ, "PYTHONPATH": str(repository / "src")},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["fasta_header"], "second")
            self.assertEqual(float(rows[0]["sequence.sequence_identity"]), 1.0)
            self.assertIn("2 scoring workers", result.stderr)
            self.assertEqual(list((root / "scratch").iterdir()), [])

    def test_batches_obey_size_and_padding_budget(self):
        sequences = tuple(
            ProteinSequence(str(i), "A" * length)
            for i, length in enumerate((19, 2, 3, 4, 1))
        )
        batches = list(embedding_batches(sequences, 3, 12))
        self.assertEqual(sum(map(len, batches)), len(sequences))
        self.assertEqual(len({s.id for batch in batches for s in batch}), len(sequences))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * max(len(s.sequence) + 2 for s in batch), 12)

    def test_cached_serial_and_spawn_workers_match_original_scores(self):
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
                references, noisy, models=("reference-deterministic-v1",),
                device="cpu", precision="auto",
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
                            request, references, noisy, root, workers=workers,
                            batch_size=1, max_batch_tokens=12, torch_threads=1,
                            cache_dir=root,
                        )
                    self.assertEqual(len(instances), 5)
                    self.assertTrue(all(adapter is instances[0] for adapter in instances))
                    self.assertEqual(actual, expected)
                    self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
