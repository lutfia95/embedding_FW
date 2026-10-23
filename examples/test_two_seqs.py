"""Compare two protein sequences with the noisy analyser's scoring methods.

Run from the repository root after installing the package:

    python examples/test_two_seqs.py
    python examples/test_two_seqs.py MKTAYIAK MKTAYIAQ --models esm2-8m

The default deterministic adapter is a software fixture. Select a PLM with
--models for biological embeddings; its checkpoint may be downloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct script execution adds the repository root for examples.* imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.score_two_sequences import score_two_sequences
from plms_framework.errors import FrameworkError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_a", nargs="?", default="MKTAYIAK")
    parser.add_argument("sequence_b", nargs="?", default="MKTAYIAQ")
    parser.add_argument(
        "--models", nargs="+", default=["reference-deterministic-v1"], metavar="MODEL"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "bf16"), default="auto"
    )
    args = parser.parse_args(argv)

    try:
        scores = score_two_sequences(
            args.sequence_a,
            args.sequence_b,
            models=args.models,
            device=args.device,
            precision=args.precision,
        )
    except FrameworkError as exc:
        parser.exit(1, f"{exc}\n")

    print(json.dumps(scores, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
