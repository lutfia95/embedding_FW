import unittest

from plms_framework.diagnostics import diagnose_embeddings
from plms_framework.errors import ValidationError
from plms_framework.external import build_imported_space
from plms_framework.models import DiagnosticSpec, RepresentationMethod


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        imported = build_imported_space(
            space_name="geometry-fixture",
            representations=[
                {
                    "method": "mean",
                    "embeddings": [
                        {"sequence_id": "a", "values": [1, 0]},
                        {"sequence_id": "b", "values": [1, 0]},
                        {"sequence_id": "c", "values": [0, 1]},
                    ],
                }
            ],
        )
        self.outputs = imported["models"]
        self.order = imported["sequence_order"]

    def test_geometry_and_hubness(self):
        result = diagnose_embeddings(
            self.outputs,
            self.order,
            DiagnosticSpec(sample_pairs=100, hubness_k=1),
        )
        model = result["models"]["geometry-fixture"]
        self.assertTrue(result["available"])
        self.assertEqual(result["sample_pair_count"], 3)
        self.assertEqual(model["embedding_dimension"], 2)
        self.assertEqual(model["norms"]["zero_count"], 0)
        self.assertEqual(model["duplicates"]["groups"], [["a", "b"]])
        self.assertAlmostEqual(model["anisotropy"]["average_pairwise_cosine"], 1 / 3)
        self.assertAlmostEqual(model["dimensions"]["participation_ratio"], 2.0)
        self.assertAlmostEqual(model["hubness"]["robin_hood_index"], 1 / 3)
        self.assertEqual(model["hubness"]["top_hubs"][0]["sequence_id"], "a")

    def test_pair_sampling_is_reproducible(self):
        records = [
            {"sequence_id": str(index), "values": [index + 1, index % 3]} for index in range(20)
        ]
        imported = build_imported_space(
            space_name="sample-fixture",
            representations=[{"method": "mean", "embeddings": records}],
        )
        spec = DiagnosticSpec(sample_pairs=17, hubness_k=2, seed=123)
        first = diagnose_embeddings(imported["models"], imported["sequence_order"], spec)
        second = diagnose_embeddings(imported["models"], imported["sequence_order"], spec)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_pair_count"], 17)

    def test_hubness_limit_is_explicit(self):
        result = diagnose_embeddings(
            self.outputs,
            self.order,
            DiagnosticSpec(hubness_k=1, max_exact_hubness_sequences=2),
        )
        hubness = result["models"]["geometry-fixture"]["hubness"]
        self.assertFalse(hubness["available"])
        self.assertIn("limited to 2", hubness["reason"])

    def test_residue_diagnostics_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "mean, max, or native"):
            diagnose_embeddings(
                self.outputs,
                self.order,
                DiagnosticSpec(representation=RepresentationMethod.RESIDUE),
            )


if __name__ == "__main__":
    unittest.main()
