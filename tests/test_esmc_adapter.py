import sys
import types
import unittest
from unittest.mock import patch

from plms_framework.adapters.esmc import _version_tuple
from plms_framework.models import Availability
from plms_framework.registry import default_registry


class _FakeModel:
    call = None

    @classmethod
    def from_pretrained(cls, artifact, **kwargs):
        cls.call = (artifact, kwargs)
        return cls()

    def eval(self):
        return self

    def parameters(self):
        yield types.SimpleNamespace(dtype="torch.float32")


class _FakeTokenizer:
    pass


class ESMCAdapterTests(unittest.TestCase):
    def test_version_comparison_distinguishes_fixed_post_release(self):
        self.assertLess(_version_tuple("3.4.1"), _version_tuple("3.4.1.post1"))
        self.assertGreater(_version_tuple("3.4.2"), _version_tuple("3.4.1.post1"))

    def test_load_uses_current_biohub_api_and_portable_attention(self):
        adapter = default_registry.adapter("esmc-300m")
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        fake_esm = types.ModuleType("esm")
        fake_models = types.ModuleType("esm.models")
        fake_esmc = types.ModuleType("esm.models.esmc")
        fake_esmc.EsmcForMaskedLM = _FakeModel
        fake_esmc.EsmcTokenizer = _FakeTokenizer

        with (
            patch.object(adapter, "availability", return_value=Availability(True)),
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "esm": fake_esm,
                    "esm.models": fake_models,
                    "esm.models.esmc": fake_esmc,
                },
            ),
            patch(
                "plms_framework.adapters.esmc.importlib.metadata.version",
                return_value="3.4.1.post1",
            ),
        ):
            adapter.load(device="cpu", precision="float32")

        artifact, kwargs = _FakeModel.call
        self.assertEqual(artifact, "biohub/ESMC-300M")
        self.assertEqual(kwargs["revision"], "main")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["dtype"], "float32")
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(adapter.provenance()["esm_version"], "3.4.1.post1")


if __name__ == "__main__":
    unittest.main()
