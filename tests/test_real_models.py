import os
import unittest

from plms_framework import Pipeline
from plms_framework.registry import default_registry
from plms_framework.serialization import request_from_dict


@unittest.skipUnless(
    os.environ.get("PLMS_RUN_MODEL_TESTS") == "1",
    "Set PLMS_RUN_MODEL_TESTS=1 to run checkpoint-download tests",
)
class RealModelTests(unittest.TestCase):
    def test_esm2_8m_shapes_and_batch_invariance(self):
        base = {
            "models": ["esm2-8m"],
            "sequences": [
                {"id": "a", "sequence": "MKTAYI"},
                {"id": "b", "sequence": "GGGGGG"},
            ],
            "representations": [
                {"method": "mean", "normalize": "none"},
                {"method": "residue", "normalize": "none"},
            ],
            "execution": {"device": "cpu", "allow_partial": False},
        }
        batched = Pipeline().embed(request_from_dict(base, compare=False))
        output = batched["models"]["esm2-8m"]
        means = [item for item in output["representations"] if item["method"] == "mean"]
        residues = [item for item in output["representations"] if item["method"] == "residue"]
        self.assertEqual(means[0]["shape"], (320,))
        self.assertEqual(residues[0]["shape"], (6, 320))

        single = dict(base)
        single["sequences"] = [base["sequences"][0]]
        alone = Pipeline().embed(request_from_dict(single, compare=False))
        alone_mean = next(
            item
            for item in alone["models"]["esm2-8m"]["representations"]
            if item["method"] == "mean"
        )
        for batched_value, alone_value in zip(
            means[0]["values"], alone_mean["values"], strict=True
        ):
            self.assertAlmostEqual(batched_value, alone_value, places=5)


@unittest.skipUnless(
    os.environ.get("PLMS_RUN_EXTENDED_MODEL_TESTS") == "1",
    "Set PLMS_RUN_EXTENDED_MODEL_TESTS=1 to run the extended checkpoint suite",
)
class ExtendedRealModelTests(unittest.TestCase):
    def test_new_checkpoint_shapes(self):
        expected_dimensions = {
            "esm1b-650m": 1280,
            "esm1v-650m-1": 1280,
            "esm1v-650m-2": 1280,
            "esm1v-650m-3": 1280,
            "esm1v-650m-4": 1280,
            "esm1v-650m-5": 1280,
            "protbert": 1024,
            "prott5-xl-bfd": 1024,
            "protalbert": 4096,
            "protxlnet": 1024,
            "progen2-small": 1024,
            "progen2-medium": 1536,
        }
        device = os.environ.get("PLMS_TEST_DEVICE", "auto")
        precision = os.environ.get("PLMS_TEST_PRECISION", "auto")
        requested = {
            item.strip()
            for item in os.environ.get("PLMS_TEST_MODELS", "").split(",")
            if item.strip()
        }
        selected = {default_registry.resolve_id(item) for item in requested}
        unsupported = selected - expected_dimensions.keys()
        self.assertFalse(
            unsupported,
            f"PLMS_TEST_MODELS entries are outside this suite: {sorted(unsupported)}",
        )
        for model_id, dimension in expected_dimensions.items():
            if selected and model_id not in selected:
                continue
            with self.subTest(model=model_id):
                request = request_from_dict(
                    {
                        "models": [model_id],
                        "sequences": [{"id": "a", "sequence": "MKTAYI"}],
                        "representations": [
                            {"method": "mean", "normalize": "none"},
                            {"method": "residue", "normalize": "none"},
                        ],
                        "execution": {
                            "device": device,
                            "precision": precision,
                            "allow_partial": False,
                        },
                    },
                    compare=False,
                )
                result = Pipeline().embed(request)
                output = next(iter(result["models"].values()))
                mean = next(
                    item for item in output["representations"] if item["method"] == "mean"
                )
                residue = next(
                    item for item in output["representations"] if item["method"] == "residue"
                )
                self.assertEqual(mean["shape"], (dimension,))
                self.assertEqual(residue["shape"], (6, dimension))
                self.assertEqual(
                    output["manifest"]["conformance"],
                    "adapter_implemented_checkpoint_validation_required",
                )


@unittest.skipUnless(
    os.environ.get("PLMS_RUN_ESMC_TESTS") == "1",
    "Set PLMS_RUN_ESMC_TESTS=1 to run the ESM-C checkpoint test",
)
class RealESMCTests(unittest.TestCase):
    def test_esmc_300m_cpu_shapes(self):
        request = request_from_dict(
            {
                "models": ["esmc-300m"],
                "sequences": [{"id": "a", "sequence": "MKTAYI"}],
                "representations": [
                    {"method": "mean", "normalize": "none"},
                    {"method": "residue", "normalize": "none"},
                ],
                "execution": {
                    "device": "cpu",
                    "precision": "float32",
                    "allow_partial": False,
                },
            },
            compare=False,
        )
        result = Pipeline().embed(request)
        output = result["models"]["esmc-300m"]
        mean = next(item for item in output["representations"] if item["method"] == "mean")
        residue = next(
            item for item in output["representations"] if item["method"] == "residue"
        )
        self.assertEqual(mean["shape"], (960,))
        self.assertEqual(residue["shape"], (6, 960))


if __name__ == "__main__":
    unittest.main()
