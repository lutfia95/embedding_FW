from __future__ import annotations

from collections.abc import Callable, Iterator

from .adapters import DeterministicAdapter, HuggingFaceAdapter, ModelAdapter
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
        ambiguous_residues="Passed to the upstream tokenizer; unsupported tokens cause validation failure",
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
    for model_id, artifact, parameters in (
        ("esmc-300m", "Biohub/ESMC-300M", 300_000_000),
        ("esmc-600m", "Biohub/ESMC-600M", 600_000_000),
        ("esmc-6b", "Biohub/ESMC-6B", 6_000_000_000),
    ):
        registry.register(
            _hf_manifest(
                id=model_id,
                display_name=artifact.rsplit("/", 1)[1],
                family="ESM-C",
                artifact=artifact,
                license="See Biohub/ESM upstream model card for this checkpoint",
                layers=None,
                dimension=None,
                parameters=parameters,
                max_length=None,
                limitations=("Checkpoint may require accepting upstream access and license terms",),
            ),
            HuggingFaceAdapter,
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
    ):
        registry.register(manifest, HuggingFaceAdapter)
    return registry


default_registry = build_default_registry()
