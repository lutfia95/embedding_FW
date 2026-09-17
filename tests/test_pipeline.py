import unittest
from dataclasses import replace

from plms_framework import Pipeline
from plms_framework.adapters import DeterministicAdapter
from plms_framework.errors import ResourceLimitError, ValidationError
from plms_framework.registry import ModelRegistry, default_registry
from plms_framework.serialization import request_from_dict


def two_model_registry():
    registry = ModelRegistry()
    reference = default_registry.manifest("reference")
    registry.register(
        replace(reference, id="reference-a", aliases=(), embedding_dimension=8),
        DeterministicAdapter,
    )
    registry.register(
        replace(reference, id="reference-b", aliases=(), embedding_dimension=16),
        DeterministicAdapter,
    )
    return registry


class PipelineTests(unittest.TestCase):
    def test_end_to_end_compare(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [
                    {"id": "a", "sequence": "MKTAYIAK"},
                    {"id": "b", "sequence": "MKTAYVAK"},
                    {"id": "c", "sequence": "GGGGGGGG"},
                ],
                "representations": [
                    {"method": "mean"},
                    {"method": "residue", "normalize": "none"},
                ],
                "metrics": [
                    "cosine_similarity",
                    "sequence_identity",
                    "blosum62_global",
                    "residue_local_alignment",
                    "residue_optimal_transport",
                ],
                "visualizations": ["residue_similarity_map"],
            },
            compare=True,
        )
        result = Pipeline().compare(request)
        output = result["models"]["reference-deterministic-v1"]
        self.assertIn("mean@model_default:cosine_similarity", output["matrices"])
        self.assertIn("blosum62_global", result["sequence_baselines"])
        self.assertEqual(len(output["residue_pairs"]), 3)
        self.assertIn("similarity_matrix", output["residue_pairs"]["a::b"])
        self.assertFalse(result["failures"])

    def test_model_validation_failure_is_partial(self):
        request = request_from_dict(
            {
                "models": ["esm2-8m", "reference"],
                "sequences": [{"id": "a", "sequence": "M" * 1023}],
            },
            compare=False,
        )
        result = Pipeline().embed(request)
        self.assertIn("reference-deterministic-v1", result["models"])
        self.assertEqual(result["failures"][0]["model"], "esm2-8m")

    def test_exact_limit(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [
                    {"id": "a", "sequence": "M"},
                    {"id": "b", "sequence": "K"},
                    {"id": "c", "sequence": "T"},
                ],
                "execution": {"max_exact_sequences": 2},
            },
            compare=True,
        )
        with self.assertRaises(ResourceLimitError):
            Pipeline().compare(request)

    def test_unimplemented_execution_modes_are_rejected(self):
        for option, value in (
            ("all_to_all", "approximate"),
            ("on_long_sequence", "window"),
            ("device", "gpu"),
            ("precision", "float64"),
        ):
            with self.subTest(option=option):
                request = request_from_dict(
                    {
                        "models": ["reference"],
                        "sequences": [{"id": "a", "sequence": "M"}],
                        "execution": {option: value},
                    },
                    compare=False,
                )
                with self.assertRaises(ValidationError):
                    Pipeline().embed(request)

    def test_invalid_string_layer_is_a_structured_partial_failure(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [{"id": "a", "sequence": "M"}],
                "representations": [{"method": "mean", "layer": "last"}],
            },
            compare=False,
        )
        result = Pipeline().embed(request)
        self.assertFalse(result["models"])
        self.assertEqual(result["failures"][0]["code"], "unsupported_capability")

    def test_duplicate_representation_key_is_rejected(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [{"id": "a", "sequence": "M"}],
                "representations": [
                    {"method": "mean", "normalize": "none"},
                    {"method": "mean", "normalize": "l2"},
                ],
            },
            compare=False,
        )
        with self.assertRaises(ValidationError):
            Pipeline().embed(request)

    def test_result_order_and_baselines_use_canonical_input(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [
                    {"id": " a ", "sequence": "mk\tt"},
                    {"id": "b", "sequence": "M K T"},
                ],
                "metrics": ["sequence_identity"],
            },
            compare=True,
        )
        result = Pipeline().compare(request)
        self.assertEqual(result["sequence_order"], ["a", "b"])
        self.assertEqual(result["sequence_baselines"]["sequence_identity"][0][1], 1.0)

    def test_residue_metric_requires_residue_representation(self):
        request = request_from_dict(
            {
                "models": ["reference"],
                "sequences": [{"id": "a", "sequence": "M"}, {"id": "b", "sequence": "K"}],
                "representations": [{"method": "mean"}],
                "metrics": ["residue_local_alignment"],
            },
            compare=True,
        )
        with self.assertRaises(ValidationError):
            Pipeline().compare(request)

    def test_pipeline_integrates_cross_model_comparison(self):
        request = request_from_dict(
            {
                "models": ["reference-a", "reference-b"],
                "sequences": [
                    {"id": "a", "sequence": "MKTAYI"},
                    {"id": "b", "sequence": "MKTAVI"},
                    {"id": "c", "sequence": "GGGGGG"},
                ],
                "representations": [{"method": "mean"}],
                "metrics": ["cosine_similarity"],
                "cross_model": {
                    "enabled": True,
                    "methods": ["matrix_spearman", "linear_cka", "consensus_rank"],
                },
            },
            compare=True,
        )
        result = Pipeline(two_model_registry()).compare(request)
        self.assertTrue(result["cross_model"]["available"])
        self.assertTrue(result["cross_model"]["comparisons"]["matrix_spearman"]["available"])
        self.assertEqual(result["model_agreement"]["model_order"], ["reference-a", "reference-b"])

    def test_pipeline_allows_single_sequence_residue_cross_model(self):
        request = request_from_dict(
            {
                "models": ["reference-a", "reference-b"],
                "sequences": [{"id": "a", "sequence": "MKTAYI"}],
                "representations": [{"method": "residue", "normalize": "none"}],
                "metrics": [],
                "cross_model": {
                    "enabled": True,
                    "methods": ["residue_cka", "residue_rsa"],
                },
            },
            compare=True,
        )
        result = Pipeline(two_model_registry()).compare(request)
        self.assertTrue(
            result["cross_model"]["comparisons"]["residue_cka"]["by_sequence"]["a"]["available"]
        )


if __name__ == "__main__":
    unittest.main()
