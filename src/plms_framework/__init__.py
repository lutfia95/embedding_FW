"""Universal protein language model representation framework."""

__version__ = "0.1.0"

from .models import (
    AnnotationAnalysisSpec,
    CompareRequest,
    CrossModelMethod,
    CrossModelSpec,
    DiagnosticSpec,
    EmbeddingRequest,
    ExecutionOptions,
    ProteinSequence,
    RepresentationSpec,
    TransformationMethod,
    TransformationSpec,
)
from .pipeline import Pipeline
from .registry import default_registry

__all__ = [
    "AnnotationAnalysisSpec",
    "CompareRequest",
    "CrossModelMethod",
    "CrossModelSpec",
    "DiagnosticSpec",
    "EmbeddingRequest",
    "ExecutionOptions",
    "Pipeline",
    "ProteinSequence",
    "RepresentationSpec",
    "TransformationMethod",
    "TransformationSpec",
    "default_registry",
]
