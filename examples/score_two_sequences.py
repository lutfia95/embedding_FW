"""Score two protein sequence strings using the noisy analyser's metrics.

Example function call (the first run may download model checkpoints)::

    from examples.score_two_sequences import score_two_sequences

    scores = score_two_sequences(
        "MKTAYIAK", "MKTAYIAQ",
        models=("esm2-8m", "esmc-300m"),
        device="auto",
    )
    print(scores)
    print(scores["esm2-8m.mean.cosine_similarity"])

Command line::

    python examples/score_two_sequences.py MKTAYIAK MKTAYIAQ --models esm2-8m

For a quick check without downloading a model, select reference-deterministic-v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from plms_framework import Pipeline
from plms_framework.errors import FrameworkError, ValidationError
from plms_framework.serialization import request_from_dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.noisy_seq_analyser import add_model_scores, index_representations, sequence_scores


def score_two_sequences(
    sequence_a: str,
    sequence_b: str,
    *,
    models: Sequence[str] = ("esm2-8m",),
    device: str = "auto",
    precision: str = "auto",
) -> dict[str, float]:
    """Embed exactly two raw sequences and return a flat dictionary of scores.

    Uses L2-normalized mean, max, and residue embeddings, with all five
    sequence metrics and fourteen embedding metrics per model from
    noisy_seq_analyser.py. Whitespace and lowercase residues are accepted.
    Model keys use the framework's resolved model IDs. Invalid inputs or
    model failures raise framework errors; partial results are disabled.
    """
    if not isinstance(sequence_a, str) or not isinstance(sequence_b, str):
        raise ValidationError("Both sequences must be strings")
    if isinstance(models, str) or not models:
        raise ValidationError("models must be a non-empty list or tuple of model IDs")
    request = request_from_dict(
        {
            "models": list(models),
            "sequences": [
                {"id": "reference", "sequence": sequence_a},
                {"id": "target", "sequence": sequence_b},
            ],
            "representations": [
                {"method": method, "normalize": "l2"}
                for method in ("mean", "max", "residue")
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
    embedded = Pipeline().embed(request)
    scores = sequence_scores(sequence_a, sequence_b)
    for model_id, output in embedded["models"].items():
        add_model_scores(
            scores,
            model_id=model_id,
            representations=index_representations(output),
            reference_id="reference",
            target_id="target",
        )
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_a", help="First raw protein sequence")
    parser.add_argument("sequence_b", help="Second raw protein sequence")
    parser.add_argument("--models", nargs="+", default=["esm2-8m"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "bf16"), default="auto"
    )
    args = parser.parse_args()
    try:
        scores = score_two_sequences(
            args.sequence_a, args.sequence_b,
            models=args.models, device=args.device, precision=args.precision,
        )
    except FrameworkError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
