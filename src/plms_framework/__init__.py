"""Universal protein language model representation framework."""

__version__ = "0.1.0"

from .models import (
    CompareRequest,
    CrossModelMethod,
    CrossModelSpec,
    DiagnosticSpec,
    EmbeddingRequest,
    ExecutionOptions,
    ProteinSequence,
    RepresentationSpec,
)
from .pipeline import Pipeline
from .registry import default_registry

__all__ = [
    "CompareRequest",
    "CrossModelMethod",
    "CrossModelSpec",
    "DiagnosticSpec",
    "EmbeddingRequest",
    "ExecutionOptions",
    "Pipeline",
    "ProteinSequence",
    "RepresentationSpec",
    "default_registry",
]
