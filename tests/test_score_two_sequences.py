from pathlib import Path

import pytest

from examples.noisy_seq_analyser import NoisySequence, build_embedding_request, collect_rows
from examples.score_two_sequences import score_two_sequences
from plms_framework import Pipeline
from plms_framework.errors import ValidationError
from plms_framework.models import ProteinSequence


@pytest.mark.parametrize("target", ["MKTAYIAK", "MKTAYIAQ", "MKTAY"])
def test_scores_match_noisy_analyser(target):
    models = ("reference-deterministic-v1",)
    references = (ProteinSequence("protein", "MKTAYIAK"),)
    noisy = [NoisySequence(Path("noisy.fasta"), "noisy_1", "protein", target)]
    request = build_embedding_request(
        references, noisy, models=models, device="cpu", precision="auto"
    )
    row = collect_rows(references, noisy, Pipeline().embed(request), Path("."))[0]
    expected = {key: value for key, value in row.items() if "." in key}

    scores = score_two_sequences("mktay iak\n", target, models=models, device="cpu")

    assert len(scores) == 19
    assert scores == pytest.approx(expected)


@pytest.mark.parametrize(
    "first,second,models",
    [
        ("", "MKTAY", ("reference-deterministic-v1",)),
        ("MKTAY", "?", ("reference-deterministic-v1",)),
        (None, "MKTAY", ("reference-deterministic-v1",)),
        ("MKTAY", "MKTAY", ()),
        ("MKTAY", "MKTAY", "esm2-8m"),
    ],
)
def test_invalid_inputs(first, second, models):
    with pytest.raises(ValidationError):
        score_two_sequences(first, second, models=models)
