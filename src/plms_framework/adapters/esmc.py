from __future__ import annotations

import importlib.util

from ..errors import DependencyUnavailableError, ModelUnavailableError
from ..models import Availability, ProteinSequence, RawEmbedding
from .base import ModelAdapter


class ESMCAdapter(ModelAdapter):
    """Lazy adapter for the official EvolutionaryScale ESM-C SDK."""

    def __init__(self, manifest) -> None:
        super().__init__(manifest)
        self._client = None
        self._device = None

    def availability(self) -> Availability:
        if importlib.util.find_spec("esm") is None or importlib.util.find_spec("torch") is None:
            return Availability(False, "dependency_missing", "Install plms-framework[esmc]")
        return Availability(True)

    def load(self, *, device: str = "auto", precision: str = "auto") -> None:
        if not self.availability().available:
            raise DependencyUnavailableError("The official ESM package and PyTorch are required")
        if self._client is not None:
            return
        import torch
        from esm.models.esmc import ESMC

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        try:
            self._client = ESMC.from_pretrained(self.manifest.artifact).to(device)
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load {self.manifest.id}: {exc}") from exc
        self._device = device

    def unload(self) -> None:
        self._client = None
        self._device = None

    def embed(
        self,
        sequences: tuple[ProteinSequence, ...],
        *,
        layer="model_default",
        device="auto",
        precision="auto",
    ):
        self.load(device=device, precision=precision)
        from esm.sdk.api import ESMProtein, LogitsConfig

        results = []
        for item in sequences:
            protein = ESMProtein(sequence=item.sequence)
            tensor = self._client.encode(protein)
            output = self._client.logits(
                tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            embeddings = output.embeddings
            # Official ESM SDK embeddings include boundary tokens.
            residue_tensor = embeddings[0, 1 : len(item.sequence) + 1].detach().float().cpu()
            results.append(
                RawEmbedding(
                    item.id,
                    residue_tensor.tolist(),
                    token_to_residue=list(range(1, len(item.sequence) + 1)),
                    metadata={"layer": "official_final", "device": self._device},
                )
            )
        return results
