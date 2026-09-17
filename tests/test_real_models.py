import os
import unittest

from plms_framework import Pipeline
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
