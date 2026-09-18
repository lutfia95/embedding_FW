import unittest

from plms_framework.adapters import ESMCAdapter, HuggingFaceAdapter
from plms_framework.registry import default_registry


class RegistryTests(unittest.TestCase):
    def test_default_registry_contains_documented_model_count(self):
        self.assertEqual(sum(1 for _ in default_registry.models()), 26)

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

    def test_requested_legacy_and_prottrans_models_are_registered(self):
        aliases = {
            "esm1b_t33_650M_UR50S": "esm1b-650m",
            "esm1v_t33_650M_UR90S_1": "esm1v-650m-1",
            "esm1v_t33_650M_UR90S_2": "esm1v-650m-2",
            "esm1v_t33_650M_UR90S_3": "esm1v-650m-3",
            "esm1v_t33_650M_UR90S_4": "esm1v-650m-4",
            "esm1v_t33_650M_UR90S_5": "esm1v-650m-5",
            "prot_bert": "protbert",
            "prott5_xl_bfd": "prott5-xl-bfd",
            "prot_albert": "protalbert",
            "prot_xlnet": "protxlnet",
            "progen2_small": "progen2-small",
            "progen2_medium": "progen2-medium",
        }
        for alias, model_id in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(default_registry.resolve_id(alias), model_id)
                self.assertIsInstance(default_registry.adapter(alias), HuggingFaceAdapter)

    def test_new_models_are_explicitly_unvalidated(self):
        for model_id in (
            "esm1b-650m",
            "esm1v-650m-1",
            "protbert",
            "prott5-xl-bfd",
            "protalbert",
            "protxlnet",
            "progen2-small",
            "progen2-medium",
        ):
            with self.subTest(model=model_id):
                manifest = default_registry.manifest(model_id)
                self.assertEqual(
                    manifest.conformance,
                    "adapter_implemented_checkpoint_validation_required",
                )

    def test_progen2_declares_causal_remote_code_boundary(self):
        manifest = default_registry.manifest("progen2_small")
        self.assertEqual(manifest.architecture, "causal")
        self.assertTrue(manifest.trust_remote_code)
        self.assertNotEqual(manifest.revision, "main")

    def test_prottrans_input_is_space_separated_without_silent_replacement(self):
        adapter = default_registry.adapter("protbert")
        self.assertEqual(adapter._model_input("AUZOB"), "A U Z O B")


if __name__ == "__main__":
    unittest.main()
