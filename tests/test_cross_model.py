import unittest

from plms_framework.cross_model import compare_cross_model, linear_cka
from plms_framework.models import CrossModelMethod, CrossModelSpec


def synthetic_outputs():
    sequence_vectors = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [-1.0, 0.0],
        "d": [0.0, -1.0],
    }
    transformed = {
        key: [-2 * value[1], 2 * value[0], 0.0] for key, value in sequence_vectors.items()
    }
    residue_one = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    residue_two = [[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [2.0, 0.0, 0.0]]

    def output(vectors, residues):
        return {
            "representations": [
                *[
                    {
                        "sequence_id": sequence_id,
                        "method": "mean",
                        "layer": "model_default",
                        "values": vector,
                    }
                    for sequence_id, vector in vectors.items()
                ],
                *[
                    {
                        "sequence_id": sequence_id,
                        "method": "residue",
                        "layer": "model_default",
                        "values": residues,
                    }
                    for sequence_id in vectors
                ],
            ]
        }

    return {
        "model-a": output(sequence_vectors, residue_one),
        "model-b": output(transformed, residue_two),
    }


class CrossModelTests(unittest.TestCase):
    def test_linear_cka_handles_different_dimensions(self):
        left = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        right = [[0, 2, 0], [-2, 0, 0], [0, -2, 0], [2, 0, 0]]
        self.assertAlmostEqual(linear_cka(left, right), 1.0)

    def test_complete_multi_sequence_comparison(self):
        methods = tuple(CrossModelMethod)
        result = compare_cross_model(
            synthetic_outputs(),
            ["a", "b", "c", "d"],
            CrossModelSpec(
                enabled=True,
                methods=methods,
                neighbor_k=(1, 2),
                max_disagreements=4,
            ),
        )
        comparisons = result["comparisons"]
        self.assertAlmostEqual(comparisons["matrix_spearman"]["values"][0][1], 1.0)
        self.assertAlmostEqual(comparisons["matrix_pearson"]["values"][0][1], 1.0)
        self.assertAlmostEqual(comparisons["linear_cka"]["values"][0][1], 1.0)
        self.assertTrue(comparisons["neighbor_overlap"]["available"])
        self.assertTrue(comparisons["consensus_rank"]["available"])
        self.assertTrue(comparisons["pair_disagreement"]["available"])
        self.assertEqual(
            result["scientific_guardrail"],
            "No raw vectors from different PLMs were directly compared",
        )

    def test_single_sequence_residue_comparison(self):
        outputs = synthetic_outputs()
        for output in outputs.values():
            output["representations"] = [
                item for item in output["representations"] if item["sequence_id"] == "a"
            ]
        result = compare_cross_model(
            outputs,
            ["a"],
            CrossModelSpec(
                enabled=True,
                methods=(CrossModelMethod.RESIDUE_CKA, CrossModelMethod.RESIDUE_RSA),
            ),
        )
        cka = result["comparisons"]["residue_cka"]["by_sequence"]["a"]
        rsa = result["comparisons"]["residue_rsa"]["by_sequence"]["a"]
        self.assertTrue(cka["available"])
        self.assertTrue(rsa["available"])
        self.assertAlmostEqual(cka["values"][0][1], 1.0)
        self.assertAlmostEqual(rsa["values"][0][1], 1.0)

    def test_matrix_correlation_reports_small_sample(self):
        result = compare_cross_model(
            synthetic_outputs(),
            ["a", "b"],
            CrossModelSpec(enabled=True),
        )
        comparison = result["comparisons"]["matrix_spearman"]
        self.assertFalse(comparison["available"])
        self.assertIn("three sequences", comparison["reason"])


if __name__ == "__main__":
    unittest.main()
