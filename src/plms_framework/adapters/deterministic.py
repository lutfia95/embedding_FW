from __future__ import annotations

import math

from ..models import Availability, ProteinSequence, RawEmbedding
from .base import ModelAdapter


class DeterministicAdapter(ModelAdapter):
    """Dependency-free reference adapter for tests, demos, and API validation.

    It is deliberately not presented as a biological PLM.
    """

    def availability(self) -> Availability:
        return Availability(True)

    def embed(
        self,
        sequences: tuple[ProteinSequence, ...],
        *,
        layer: str | int = "model_default",
        device: str = "auto",
        precision: str = "auto",
    ) -> list[RawEmbedding]:
        dimension = self.manifest.embedding_dimension or 8
        results = []
        for item in sequences:
            residues = []
            for position, amino_acid in enumerate(item.sequence):
                code = ord(amino_acid) - 64
                vector = [
                    math.sin((code + 1) * (axis + 1) * 0.17 + position * 0.11)
                    + math.cos((code + position + 1) * (axis + 1) * 0.07)
                    for axis in range(dimension)
                ]
                residues.append(vector)
            results.append(
                RawEmbedding(
                    sequence_id=item.id,
                    residues=residues,
                    token_to_residue=list(range(1, len(item.sequence) + 1)),
                    metadata={"reference_only": True, "layer": layer},
                )
            )
        return results
