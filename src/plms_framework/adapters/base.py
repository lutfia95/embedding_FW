from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..errors import CapabilityError, ResourceLimitError
from ..models import Availability, ModelManifest, ProteinSequence, RawEmbedding, RepresentationSpec


class ModelAdapter(ABC):
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def availability(self) -> Availability:
        raise NotImplementedError

    def validate_request(
        self, sequences: Iterable[ProteinSequence], representations: Iterable[RepresentationSpec]
    ) -> None:
        for spec in representations:
            if spec.method not in self.manifest.representations:
                raise CapabilityError(
                    f"{self.manifest.id} does not support representation {spec.method}",
                    context={"model": self.manifest.id, "representation": str(spec.method)},
                )
            if (
                isinstance(spec.layer, int)
                and self.manifest.num_layers is not None
                and not 0 <= spec.layer <= self.manifest.num_layers
            ):
                raise CapabilityError(f"Layer {spec.layer} is invalid for {self.manifest.id}")
        if self.manifest.max_native_length is not None:
            too_long = [
                s.id for s in sequences if len(s.sequence) > self.manifest.max_native_length
            ]
            if too_long:
                raise ResourceLimitError(
                    f"Sequences exceed native length {self.manifest.max_native_length}; truncation is forbidden",
                    context={"model": self.manifest.id, "sequence_ids": too_long},
                )

    def load(self, *, device: str = "auto", precision: str = "auto") -> None:
        return None

    def unload(self) -> None:
        return None

    @abstractmethod
    def embed(
        self,
        sequences: tuple[ProteinSequence, ...],
        *,
        layer: str | int = "model_default",
        device: str = "auto",
        precision: str = "auto",
    ) -> list[RawEmbedding]:
        raise NotImplementedError

    def estimate_resources(self, sequences: Iterable[ProteinSequence]) -> dict:
        residues = sum(len(sequence.sequence) for sequence in sequences)
        dimension = self.manifest.embedding_dimension or 0
        return {
            "model": self.manifest.id,
            "residues": residues,
            "embedding_bytes_float32": residues * dimension * 4,
            "parameters": self.manifest.parameters,
            "checkpoint_bytes_estimate": (self.manifest.parameters or 0) * 2,
        }

    def provenance(self) -> dict:
        return {
            "model_id": self.manifest.id,
            "artifact": self.manifest.artifact,
            "requested_revision": self.manifest.revision,
            "adapter_version": self.manifest.adapter_version,
            "backend": self.manifest.backend,
        }
