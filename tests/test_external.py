import math
import unittest

from plms_framework.errors import ResourceLimitError, ValidationError
from plms_framework.external import build_imported_space


class ExternalSpaceTests(unittest.TestCase):
    def test_vector_and_residue_representations(self):
        result = build_imported_space(
            space_name="lab:model-v1",
            source={"provider": "Example Lab", "revision": "abc123"},
            representations=[
                {
                    "method": "mean",
                    "layer": 12,
                    "normalization": "l2",
                    "embeddings": [
                        {"sequence_id": "a", "values": [1, 0]},
                        {"sequence_id": "b", "values": [0, 1]},
                    ],
                },
                {
                    "method": "residue",
                    "embeddings": [
                        {"sequence_id": "a", "values": [[1, 0], [0, 1]]},
                        {"sequence_id": "b", "values": [[0, 1], [1, 0], [1, 1]]},
                    ],
                },
            ],
        )

        self.assertEqual(result["sequence_order"], ["a", "b"])
        model = result["models"]["lab:model-v1"]
        self.assertEqual(model["manifest"]["revision"], "abc123")
        self.assertEqual(
            model["manifest"]["conformance"], "externally_supplied_structural_validation_only"
        )
        self.assertEqual(len(model["representations"]), 4)
        self.assertEqual(model["representations"][0]["shape"], [2])
        self.assertEqual(model["representations"][2]["shape"], [2, 2])
        self.assertEqual(result["provenance"]["scalar_value_count"], 14)
        self.assertEqual(len(result["provenance"]["payload_sha256"]), 64)

    def test_rejects_inconsistent_dimensions(self):
        with self.assertRaisesRegex(ValidationError, "dimensions must be consistent"):
            build_imported_space(
                space_name="broken",
                representations=[
                    {
                        "method": "mean",
                        "embeddings": [
                            {"sequence_id": "a", "values": [1, 2]},
                            {"sequence_id": "b", "values": [1, 2, 3]},
                        ],
                    }
                ],
            )

    def test_rejects_inconsistent_sequence_order(self):
        with self.assertRaisesRegex(ValidationError, "identical sequence IDs"):
            build_imported_space(
                space_name="broken",
                representations=[
                    {
                        "method": "mean",
                        "embeddings": [
                            {"sequence_id": "a", "values": [1]},
                            {"sequence_id": "b", "values": [2]},
                        ],
                    },
                    {
                        "method": "residue",
                        "embeddings": [
                            {"sequence_id": "b", "values": [[2]]},
                            {"sequence_id": "a", "values": [[1]]},
                        ],
                    },
                ],
            )

    def test_rejects_nonfinite_or_boolean_values(self):
        for value in (math.nan, math.inf, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                build_imported_space(
                    space_name="broken",
                    representations=[
                        {
                            "method": "mean",
                            "embeddings": [{"sequence_id": "a", "values": [value]}],
                        }
                    ],
                )

    def test_enforces_scalar_limit(self):
        with self.assertRaises(ResourceLimitError):
            build_imported_space(
                space_name="too-large",
                representations=[
                    {
                        "method": "mean",
                        "embeddings": [{"sequence_id": "a", "values": [1, 2, 3]}],
                    }
                ],
                max_scalar_values=2,
            )


if __name__ == "__main__":
    unittest.main()
