"""Check pair threads, incremental scores, and memory release without PLM downloads."""

import csv
import weakref
from dataclasses import replace
from threading import Barrier, Event, current_thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from noisy_seq_analyser import (
    NoisySequence,
    ScoreOutput,
    build_embedding_request,
    cpu_worker_adapter,
    run_batched_analysis,
)
from plms_framework import Pipeline
from plms_framework.adapters.deterministic import DeterministicAdapter
from plms_framework.errors import ResourceLimitError
from plms_framework.models import ProteinSequence
from plms_framework.registry import ModelRegistry, default_registry

MODEL = "reference-deterministic-v1"


def input_pairs(root):
    references = (ProteinSequence("first", "ACD"), ProteinSequence("second", "MKT"))
    noisy = [
        NoisySequence(root / "one.fa", "noisy_1", "first", "ACE"),
        NoisySequence(root / "two.fa", "noisy_2", "second", "MKA"),
    ]
    request = build_embedding_request(
        references, noisy, models=(MODEL,), device="cpu", precision="auto"
    )
    return request, references, noisy


def run(request, references, noisy, root, **kwargs):
    return run_batched_analysis(
        request,
        references,
        noisy,
        root,
        workers=2,
        batch_size=2,
        max_batch_tokens=20,
        torch_threads=1,
        output_path=root / "scores",
        model_wise=True,
        cache_dir=root / "cache",
        **kwargs,
    )


def read_scores(root):
    with (root / "scores" / f"{MODEL}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_pair_threads_release_embeddings_before_scores_are_flushed(tmp_path):
    request, references, noisy = input_pairs(tmp_path)
    barrier = Barrier(2)
    threads, pairs, array_refs = set(), [], {}
    original_embed = DeterministicAdapter.embed
    original_pipeline_embed = Pipeline.embed
    original_write = ScoreOutput.write

    class ArrayList(list):
        pass

    def embed(adapter, sequences, **kwargs):
        threads.add(current_thread().name)
        pairs.append(tuple(sequence.id for sequence in sequences))
        barrier.wait(timeout=10)
        return original_embed(adapter, sequences, **kwargs)

    def track_arrays(pipeline, pair_request):
        result = original_pipeline_embed(pipeline, pair_request)
        key = int(pair_request.sequences[-1].id.removeprefix("noisy_")) - 1
        refs = []
        for representation in result["models"][MODEL]["representations"]:
            representation["values"] = ArrayList(representation["values"])
            refs.append(weakref.ref(representation["values"]))
        array_refs[key] = refs
        return result

    def write(sink, pair_index, model_index, row):
        assert all(reference() is None for reference in array_refs[pair_index])
        original_write(sink, pair_index, model_index, row)
        assert len(read_scores(tmp_path)) >= 1  # Readable while the run is active.

    with (
        patch.object(DeterministicAdapter, "embed", embed),
        patch.object(Pipeline, "embed", track_arrays),
        patch.object(ScoreOutput, "write", write),
    ):
        assert run(request, references, noisy, tmp_path) == 2
    assert len(threads) == 2
    assert all(name.startswith("noisy-pair") for name in threads)
    assert set(pairs) == {("reference_000001", "noisy_1"), ("reference_000002", "noisy_2")}
    assert len(read_scores(tmp_path)) == 2
    assert list((tmp_path / "cache").iterdir()) == []
    assert not list(tmp_path.rglob("embeddings.bin"))


def test_flushed_output_survives_later_failure_and_cache_is_cleaned(tmp_path):
    request, references, noisy = input_pairs(tmp_path)
    flushed = Event()
    original_embed = DeterministicAdapter.embed
    original_write = ScoreOutput.write

    def embed(adapter, sequences, **kwargs):
        if any(sequence.id == "noisy_2" for sequence in sequences):
            assert flushed.wait(timeout=10)
            raise RuntimeError("inference failed")
        return original_embed(adapter, sequences, **kwargs)

    def write(*args):
        original_write(*args)
        flushed.set()

    with (
        patch.object(DeterministicAdapter, "embed", embed),
        patch.object(ScoreOutput, "write", write),
        pytest.raises(RuntimeError, match="inference failed"),
    ):
        run(request, references, noisy, tmp_path)
    rows = read_scores(tmp_path)
    assert len(rows) == 1
    assert rows[0]["fasta_header"] == "first"
    assert rows[0][f"{MODEL}.status"] == "ok"
    assert list((tmp_path / "cache").iterdir()) == []


def test_long_pairs_are_recorded_without_embedding_and_short_pairs_continue(tmp_path):
    request, references, noisy = input_pairs(tmp_path)
    noisy[0] = replace(noisy[0], sequence="ACDEF")
    manifest = replace(default_registry.manifest(MODEL), max_native_length=3)
    registry = ModelRegistry()
    registry.register(manifest, DeterministicAdapter)
    observed = []
    original_embed = DeterministicAdapter.embed

    def embed(adapter, sequences, **kwargs):
        observed.extend(sequence.id for sequence in sequences)
        return original_embed(adapter, sequences, **kwargs)

    with (
        patch("plms_framework.pipeline.default_registry", registry),
        patch.object(DeterministicAdapter, "embed", embed),
    ):
        run(request, references, noisy, tmp_path)
    rows = {row["fasta_header"]: row for row in read_scores(tmp_path)}
    assert rows["first"][f"{MODEL}.status"] == "skipped_long_sequence"
    assert rows["first"][f"{MODEL}.mean.cosine_similarity"] == ""
    assert "truncation is forbidden" in rows["first"][f"{MODEL}.error"]
    assert rows["second"][f"{MODEL}.status"] == "ok"
    assert "noisy_1" not in observed
    with (
        patch("plms_framework.pipeline.default_registry", registry),
        pytest.raises(ResourceLimitError, match="native length 3"),
    ):
        run(request, references, noisy, tmp_path, long_sequences="error")


def test_cpu_threads_share_weights_and_keep_independent_model_caches():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).eval()
    model.rotary_cache = ["original"]
    adapter = SimpleNamespace(_model=model, _tokenizer={"padding": []}, _device="cpu")
    first, second = cpu_worker_adapter(adapter), cpu_worker_adapter(adapter)
    assert first._model is not second._model
    assert first._model.weight is model.weight
    assert second._model.bias is model.bias
    first._model.rotary_cache.append("first")
    first._tokenizer["padding"].append("first")
    assert second._model.rotary_cache == ["original"]
    assert second._tokenizer["padding"] == []
    with torch.inference_mode():
        assert torch.equal(first._model(torch.ones(1, 2)), model(torch.ones(1, 2)))
