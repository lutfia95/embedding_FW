import unittest

from plms_framework.errors import ValidationError
from plms_framework.external import build_imported_space
from plms_framework.models import TransformationSpec
from plms_framework.transformations import transform_embeddings


class TransformationTests(unittest.TestCase):
    def setUp(self):
        self.source = build_imported_space(
            space_name="source-space",
            representations=[
                {
                    "method": "mean",
                    "normalization": "l2",
                    "embeddings": [
                        {"sequence_id": "a", "values": [1, 2]},
                        {"sequence_id": "b", "values": [3, 4]},
                        {"sequence_id": "c", "values": [5, 6]},
                    ],
                }
            ],
        )

    def test_mean_center_on_explicit_fit_subset(self):
        original = [
            item["values"][:] for item in self.source["models"]["source-space"]["representations"]
        ]
        result = transform_embeddings(
            self.source["models"],
            self.source["sequence_order"],
            TransformationSpec(fit_sequence_ids=("a", "b"), hubness_k=1),
        )

        derived = result["models"]["source-space::mean-centered"]
        values = [item["values"] for item in derived["representations"]]
        self.assertEqual(values, [[-1.0, -1.0], [1.0, 1.0], [3.0, 3.0]])
        self.assertEqual(
            result["transformation_parameters"]["source-space::mean-centered"]["mean_vector"],
            [2.0, 3.0],
        )
        self.assertEqual(result["transformation"]["fit_scope"], "explicit_subset")
        self.assertEqual(derived["representations"][0]["normalization"], "mean_centered")
        self.assertEqual(len(result["provenance"]["parameter_checksums"].popitem()[1]), 64)
        self.assertIn("before", result["diagnostics"])
        self.assertIn("after", result["diagnostics"])
        self.assertEqual(
            [item["values"] for item in self.source["models"]["source-space"]["representations"]],
            original,
        )

    def test_all_sequence_fit_centers_each_dimension(self):
        result = transform_embeddings(
            self.source["models"],
            self.source["sequence_order"],
            TransformationSpec(hubness_k=1),
        )
        values = [
            item["values"]
            for item in result["models"]["source-space::mean-centered"]["representations"]
        ]
        for column in range(2):
            self.assertAlmostEqual(sum(row[column] for row in values) / len(values), 0.0)
        self.assertEqual(result["transformation"]["fit_scope"], "all_sequences")
        self.assertTrue(result["transformation"]["warnings"])

    def test_unknown_fit_id_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "absent from the source"):
            transform_embeddings(
                self.source["models"],
                self.source["sequence_order"],
                TransformationSpec(fit_sequence_ids=("missing",)),
            )


if __name__ == "__main__":
    unittest.main()
