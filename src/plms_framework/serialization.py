from __future__ import annotations

from .errors import ValidationError
from .models import (
    CompareRequest,
    CrossModelMethod,
    CrossModelSpec,
    EmbeddingRequest,
    ExecutionOptions,
    Metric,
    Normalization,
    ProteinSequence,
    Region,
    RepresentationMethod,
    RepresentationSpec,
    Visualization,
)


def request_from_dict(data: dict, *, compare: bool) -> EmbeddingRequest | CompareRequest:
    try:
        sequences = tuple(
            ProteinSequence(
                id=item["id"],
                sequence=item["sequence"],
                metadata=dict(item.get("metadata", {})),
                regions=tuple(Region(**region) for region in item.get("regions", [])),
            )
            for item in data["sequences"]
        )
        representations = tuple(
            RepresentationSpec(
                method=RepresentationMethod(item.get("method", "mean")),
                layer=item.get("layer", "model_default"),
                normalize=Normalization(item.get("normalize", "l2")),
                window_size=item.get("window_size"),
                window_step=item.get("window_step"),
            )
            for item in data.get("representations", [{"method": "mean"}])
        )
        execution = ExecutionOptions(**data.get("execution", {}))
        common = {
            "models": tuple(data["models"]),
            "sequences": sequences,
            "representations": representations,
            "execution": execution,
        }
        if compare:
            cross_data = data.get("cross_model", {})
            cross_model = CrossModelSpec(
                enabled=cross_data.get("enabled", False),
                methods=tuple(
                    CrossModelMethod(item)
                    for item in cross_data.get("methods", ["matrix_spearman"])
                ),
                representation=RepresentationMethod(cross_data.get("representation", "mean")),
                neighbor_k=tuple(cross_data.get("neighbor_k", [1, 5, 10])),
                max_disagreements=cross_data.get("max_disagreements", 20),
            )
            return CompareRequest(
                **common,
                metrics=tuple(Metric(item) for item in data.get("metrics", ["cosine_similarity"])),
                visualizations=tuple(
                    Visualization(item) for item in data.get("visualizations", [])
                ),
                cross_model=cross_model,
            )
        return EmbeddingRequest(**common)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid request: {exc}") from exc
