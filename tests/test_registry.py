import unittest

from plms_framework.adapters import ESMCAdapter
from plms_framework.registry import default_registry


class RegistryTests(unittest.TestCase):
    def test_esmc_uses_official_adapter_with_declared_capabilities(self):
        adapter = default_registry.adapter("esmc-300m")
        manifest = adapter.manifest

        self.assertIsInstance(adapter, ESMCAdapter)
        self.assertEqual(manifest.backend, "biohub_esm")
        self.assertEqual(manifest.embedding_dimension, 960)
        self.assertEqual(manifest.num_layers, 30)
        self.assertEqual(manifest.max_native_length, 2046)
        self.assertEqual(manifest.devices, ("cpu", "cuda"))
        self.assertEqual(manifest.precisions, ("float32", "bfloat16"))

    def test_esmc_official_names_are_aliases(self):
        self.assertEqual(default_registry.resolve_id("esmc_600m"), "esmc-600m")
        self.assertEqual(default_registry.resolve_id("biohub/ESMC-6B"), "esmc-6b")
        self.assertEqual(default_registry.resolve_id("Biohub/ESMC-300M"), "esmc-300m")


if __name__ == "__main__":
    unittest.main()
