from __future__ import annotations

from collections.abc import Callable, Iterator

from .adapters import DeterministicAdapter, ESMCAdapter, HuggingFaceAdapter, ModelAdapter
from .errors import ValidationError
from .models import ModelManifest, RepresentationMethod

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
CORE_REPRESENTATIONS = (
    RepresentationMethod.RESIDUE,
    RepresentationMethod.MEAN,
    RepresentationMethod.MAX,
    RepresentationMethod.REGION_MEAN,
    RepresentationMethod.WINDOW_MEAN,
)


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[ModelManifest, Callable[[ModelManifest], ModelAdapter]]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self, manifest: ModelManifest, factory: Callable[[ModelManifest], ModelAdapter]
    ) -> None:
        if manifest.id in self._entries:
            raise ValueError(f"Duplicate model ID: {manifest.id}")
        self._entries[manifest.id] = (manifest, factory)
        for alias in manifest.aliases:
            if alias in self._aliases:
                raise ValueError(f"Duplicate model alias: {alias}")
            self._aliases[alias] = manifest.id

    def resolve_id(self, model_id: str) -> str:
        resolved = self._aliases.get(model_id, model_id)
        if resolved not in self._entries:
            raise ValidationError(f"Unknown model: {model_id}", context={"model": model_id})
        return resolved

    def manifest(self, model_id: str) -> ModelManifest:
        return self._entries[self.resolve_id(model_id)][0]

    def adapter(self, model_id: str) -> ModelAdapter:
        manifest, factory = self._entries[self.resolve_id(model_id)]
        return factory(manifest)

    def models(self) -> Iterator[tuple[ModelManifest, ModelAdapter]]:
        for manifest, factory in self._entries.values():
            yield manifest, factory(manifest)


def _hf_manifest(
    *,
    id: str,
    display_name: str,
    family: str,
    artifact: str,
    license: str,
    layers: int | None,
    dimension: int | None,
    parameters: int | None,
    max_length: int | None,
    architecture: str = "encoder",
    preprocess: str = "raw",
    aliases: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    revision: str = "main",
    conformance: str = "adapter_implemented_checkpoint_validation_required",
    trust_remote_code: bool = False,
) -> ModelManifest:
    return ModelManifest(
        id=id,
        display_name=display_name,
        family=family,
        provider=artifact.split("/")[0],
        artifact=artifact,
        revision=revision,
        license=license,
        citation="See the upstream model card and its primary publication",
        architecture=architecture,
        input_modalities=("sequence",),
        alphabet=AA_ALPHABET,
        ambiguous_residues="Passed to the upstream tokenizer, which may encode ambiguous residues as unknown",
        tokenizer_convention="One residue token per amino acid plus model-specific boundary tokens",
        max_native_length=max_length,
        num_layers=layers,
        embedding_dimension=dimension,
        representations=CORE_REPRESENTATIONS,
        default_layer="final_hidden_state",
        one_token_per_residue=True,
        special_tokens="Excluded from residue representations and deterministic pooling",
        devices=("cpu", "cuda", "mps"),
        precisions=("float32", "float16", "bfloat16"),
        parameters=parameters,
        conformance=conformance,
        limitations=limitations,
        aliases=aliases,
        backend="huggingface",
        preprocess=preprocess,
        trust_remote_code=trust_remote_code,
    )


def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelManifest(
            id="reference-deterministic-v1",
            display_name="Deterministic reference encoder (not a PLM)",
            family="reference",
            provider="plms-framework",
            artifact="built-in",
            revision="1",
            license="Apache-2.0",
            citation="Not applicable",
            architecture="reference",
            input_modalities=("sequence",),
            alphabet=AA_ALPHABET,
            ambiguous_residues="Encoded deterministically",
            tokenizer_convention="Exactly one residue per token",
            max_native_length=None,
            num_layers=0,
            embedding_dimension=16,
            representations=CORE_REPRESENTATIONS,
            default_layer=0,
            one_token_per_residue=True,
            special_tokens="None",
            devices=("cpu",),
            precisions=("float64",),
            parameters=0,
            conformance="tested",
            limitations=("For software testing only; not a biological protein language model",),
            aliases=("reference",),
            backend="built_in",
        ),
        DeterministicAdapter,
    )
    esm2 = [
        ("esm2-8m", "facebook/esm2_t6_8M_UR50D", 6, 320, 8_000_000),
        ("esm2-35m", "facebook/esm2_t12_35M_UR50D", 12, 480, 35_000_000),
        ("esm2-150m", "facebook/esm2_t30_150M_UR50D", 30, 640, 150_000_000),
        ("esm2-650m", "facebook/esm2_t33_650M_UR50D", 33, 1280, 650_000_000),
        ("esm2-3b", "facebook/esm2_t36_3B_UR50D", 36, 2560, 3_000_000_000),
        ("esm2-15b", "facebook/esm2_t48_15B_UR50D", 48, 5120, 15_000_000_000),
    ]
    for model_id, artifact, layers, dimension, parameters in esm2:
        registry.register(
            _hf_manifest(
                id=model_id,
                display_name=artifact.rsplit("/", 1)[1],
                family="ESM-2",
                artifact=artifact,
                license="MIT (verify upstream model card for the selected revision)",
                layers=layers,
                dimension=dimension,
                parameters=parameters,
                max_length=1022,
                aliases=(artifact.rsplit("/", 1)[1],),
                revision=(
                    "c731040fcd8d73dceaa04b0a8e6329b345b0f5df" if model_id == "esm2-8m" else "main"
                ),
                conformance=(
                    "tested_cpu_shapes_masking_batch_invariance"
                    if model_id == "esm2-8m"
                    else "adapter_implemented_checkpoint_validation_required"
                ),
            ),
            HuggingFaceAdapter,
        )
    for model_id, artifact, family, revision in (
        (
            "esm1b-650m",
            "facebook/esm1b_t33_650M_UR50S",
            "ESM-1b",
            "7b37824baec4d3658e1df7479222a7c79b465b76",
        ),
        (
            "esm1v-650m-1",
            "facebook/esm1v_t33_650M_UR90S_1",
            "ESM-1v",
            "8bfdb1892536cc77bd0760b9c25ddced2cd0b4c8",
        ),
        (
            "esm1v-650m-2",
            "facebook/esm1v_t33_650M_UR90S_2",
            "ESM-1v",
            "3c1e9e64480f069b163e456c361bfd80a8bab04c",
        ),
        (
            "esm1v-650m-3",
            "facebook/esm1v_t33_650M_UR90S_3",
            "ESM-1v",
            "0b00fd112e63f6b5e70a9cd8484d4e660312ce70",
        ),
        (
            "esm1v-650m-4",
            "facebook/esm1v_t33_650M_UR90S_4",
            "ESM-1v",
            "443968f644da132d323bbc6321a6d443149fa57b",
        ),
        (
            "esm1v-650m-5",
            "facebook/esm1v_t33_650M_UR90S_5",
            "ESM-1v",
            "fb2e51cb0f605cbad2c4ca3bb784be7bc8e2f4a8",
        ),
    ):
        upstream_name = artifact.rsplit("/", 1)[1]
        registry.register(
            _hf_manifest(
                id=model_id,
                display_name=upstream_name,
                family=family,
                artifact=artifact,
                license="MIT",
                layers=33,
                dimension=1280,
                parameters=650_000_000,
                max_length=1022,
                aliases=(upstream_name, artifact),
                revision=revision,
                limitations=(
                    "Large checkpoint download and substantial accelerator memory may be required",
                    "Checkpoint-specific numerical conformance has not yet been run in this repository",
                ),
            ),
            HuggingFaceAdapter,
        )
    for model_id, artifact, layers, dimension, parameters, aliases in (
        (
            "esmc-300m",
            "biohub/ESMC-300M",
            30,
            960,
            300_000_000,
            ("ESMC-300M", "esmc_300m", "biohub/ESMC-300M", "Biohub/ESMC-300M"),
        ),
        (
            "esmc-600m",
            "biohub/ESMC-600M",
            36,
            1152,
            600_000_000,
            ("ESMC-600M", "esmc_600m", "biohub/ESMC-600M", "Biohub/ESMC-600M"),
        ),
        (
            "esmc-6b",
            "biohub/ESMC-6B",
            80,
            2560,
            6_000_000_000,
            ("ESMC-6B", "esmc_6b", "biohub/ESMC-6B", "Biohub/ESMC-6B"),
        ),
    ):
        registry.register(
            ModelManifest(
                id=model_id,
                display_name=artifact.rsplit("/", 1)[1],
                family="ESM-C",
                provider="biohub",
                artifact=artifact,
                revision="main",
                license="MIT; review the Biohub model card and Acceptable Use Policy",
                citation="See the upstream model card and ESM-C publication",
                architecture="encoder",
                input_modalities=("sequence",),
                alphabet=AA_ALPHABET,
                ambiguous_residues="B, U, Z, O, and X have explicit tokenizer entries; J maps to unknown",
                tokenizer_convention="One residue token per amino acid plus <cls> and <eos>",
                max_native_length=2046,
                num_layers=layers,
                embedding_dimension=dimension,
                representations=CORE_REPRESENTATIONS,
                default_layer="final_hidden_state",
                one_token_per_residue=True,
                special_tokens="<cls>, <eos>, and padding are excluded from residue representations",
                devices=("cpu", "cuda"),
                precisions=("float32", "bfloat16"),
                parameters=parameters,
                conformance="adapter_implemented_checkpoint_validation_required",
                limitations=(
                    "Requires Python 3.12+ and the optional Biohub esm backend",
                    "Published context is 2048 tokens, leaving 2046 native residues after boundaries",
                    "Large checkpoint downloads and substantial accelerator memory may be required",
                ),
                aliases=aliases,
                backend="biohub_esm",
            ),
            ESMCAdapter,
        )
    for manifest in (
        _hf_manifest(
            id="prott5-xl-uniref50",
            display_name="ProtT5-XL-UniRef50",
            family="ProtTrans",
            artifact="Rostlab/prot_t5_xl_uniref50",
            license="See upstream model card",
            layers=24,
            dimension=1024,
            parameters=3_000_000_000,
            max_length=None,
            architecture="encoder-decoder",
            preprocess="space_separated",
            aliases=("prot_t5_xl_uniref50",),
        ),
        _hf_manifest(
            id="prott5-xl-bfd",
            display_name="ProtT5-XL-BFD",
            family="ProtTrans",
            artifact="Rostlab/prot_t5_xl_bfd",
            license="See upstream model card",
            layers=24,
            dimension=1024,
            parameters=3_000_000_000,
            max_length=None,
            architecture="encoder-decoder",
            preprocess="space_separated",
            aliases=("prott5_xl_bfd", "prot_t5_xl_bfd"),
            revision="7ae1d5c1d148d6c65c7e294cc72807e5b454fdb7",
            limitations=("Large checkpoint download and accelerator memory requirement",),
        ),
        _hf_manifest(
            id="protbert",
            display_name="ProtBert",
            family="ProtTrans",
            artifact="Rostlab/prot_bert",
            license="See upstream model card",
            layers=30,
            dimension=1024,
            parameters=420_000_000,
            max_length=None,
            preprocess="space_separated",
            aliases=("prot_bert",),
            revision="7a894481acdc12202f0a415dd567f6cfdb698908",
        ),
        _hf_manifest(
            id="protbert-bfd",
            display_name="ProtBert-BFD",
            family="ProtTrans",
            artifact="Rostlab/prot_bert_bfd",
            license="See upstream model card",
            layers=30,
            dimension=1024,
            parameters=420_000_000,
            max_length=None,
            preprocess="space_separated",
            aliases=("prot_bert_bfd",),
        ),
        _hf_manifest(
            id="protalbert",
            display_name="ProtAlbert",
            family="ProtTrans",
            artifact="Rostlab/prot_albert",
            license="See upstream repository",
            layers=12,
            dimension=4096,
            parameters=224_000_000,
            max_length=None,
            preprocess="space_separated",
            aliases=("prot_albert",),
            revision="2384f5574ea4b85218ae7d5e21d17957105e672e",
            limitations=("Upstream checkpoint has minimal model-card metadata",),
        ),
        _hf_manifest(
            id="protxlnet",
            display_name="ProtXLNet",
            family="ProtTrans",
            artifact="Rostlab/prot_xlnet",
            license="See upstream repository",
            layers=30,
            dimension=1024,
            parameters=409_000_000,
            max_length=None,
            preprocess="space_separated",
            aliases=("prot_xlnet",),
            revision="7fe4d7b13695ccf8fb14a257986976bd7f704782",
            limitations=(
                "Uses XLNet permutation-language-model architecture",
                "Upstream checkpoint has minimal model-card metadata",
            ),
        ),
        _hf_manifest(
            id="ankh-base",
            display_name="Ankh Base",
            family="Ankh",
            artifact="ElnaggarLab/ankh-base",
            license="CC-BY-NC-SA-4.0",
            layers=None,
            dimension=768,
            parameters=450_000_000,
            max_length=None,
            architecture="encoder-decoder",
        ),
        _hf_manifest(
            id="ankh-large",
            display_name="Ankh Large",
            family="Ankh",
            artifact="ElnaggarLab/ankh-large",
            license="CC-BY-NC-SA-4.0",
            layers=None,
            dimension=1536,
            parameters=1_150_000_000,
            max_length=None,
            architecture="encoder-decoder",
        ),
        _hf_manifest(
            id="progen2-small",
            display_name="ProGen2 Small",
            family="ProGen2",
            artifact="hugohrban/progen2-small",
            license="BSD-3-Clause; verify the conversion and upstream model terms",
            layers=12,
            dimension=1024,
            parameters=151_000_000,
            max_length=1022,
            architecture="causal",
            aliases=("progen2_small",),
            revision="43237a0b733c6629226a079266d2985c9fdce9b7",
            trust_remote_code=True,
            limitations=(
                "Community Hugging Face conversion of the upstream ProGen2 checkpoint",
                "Loading executes revision-pinned remote model code",
                "Causal residue representations only encode left context",
                "The 1,024-token context is conservatively exposed as 1,022 residues",
            ),
        ),
        _hf_manifest(
            id="progen2-medium",
            display_name="ProGen2 Medium",
            family="ProGen2",
            artifact="hugohrban/progen2-medium",
            license="BSD-3-Clause; verify the conversion and upstream model terms",
            layers=27,
            dimension=1536,
            parameters=764_000_000,
            max_length=1022,
            architecture="causal",
            aliases=("progen2_medium",),
            revision="6f022f76b5db11142dc8f446cfab111fe75422d6",
            trust_remote_code=True,
            limitations=(
                "Community Hugging Face conversion of the upstream ProGen2 checkpoint",
                "Loading executes revision-pinned remote model code",
                "Causal residue representations only encode left context",
                "The 1,024-token context is conservatively exposed as 1,022 residues",
            ),
        ),
    ):
        registry.register(manifest, HuggingFaceAdapter)
    return registry


default_registry = build_default_registry()
