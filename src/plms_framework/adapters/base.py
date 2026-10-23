from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..errors import CapabilityError, ResourceLimitError
from ..models import (
    Availability,
    ExecutionOptions,
    ModelManifest,
    ProteinSequence,
    RawEmbedding,
    RepresentationSpec,
)

_PRECISION_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
}


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
            if isinstance(spec.layer, bool) or not isinstance(spec.layer, (str, int)):
                raise CapabilityError(
                    f"Layer must be 'model_default' or an integer for {self.manifest.id}"
                )
            if isinstance(spec.layer, str) and spec.layer != "model_default":
                raise CapabilityError(
                    f"String layer {spec.layer!r} is invalid for {self.manifest.id}; "
                    "use 'model_default' or an integer"
                )
            if isinstance(spec.layer, int) and (
                self.manifest.num_layers is not None
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

    def validate_execution(self, execution: ExecutionOptions) -> None:
        if execution.device != "auto" and execution.device not in self.manifest.devices:
            raise CapabilityError(
                f"{self.manifest.id} does not advertise device {execution.device!r}",
                context={
                    "model": self.manifest.id,
                    "device": execution.device,
                    "supported_devices": list(self.manifest.devices),
                },
            )
        if execution.precision == "auto":
            return
        precision = _PRECISION_ALIASES[execution.precision]
        if precision not in self.manifest.precisions:
            raise CapabilityError(
                f"{self.manifest.id} does not advertise precision {execution.precision!r}",
                context={
                    "model": self.manifest.id,
                    "precision": execution.precision,
                    "supported_precisions": list(self.manifest.precisions),
                },
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
