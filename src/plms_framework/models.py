from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .errors import ValidationError


class RepresentationMethod(StrEnum):
    RESIDUE = "residue"
    MEAN = "mean"
    NATIVE = "native"
    MAX = "max"
    REGION_MEAN = "region_mean"
    WINDOW_MEAN = "window_mean"


class Normalization(StrEnum):
    NONE = "none"
    L2 = "l2"


class ImportedNormalization(StrEnum):
    UNKNOWN = "unknown"
    NONE = "none"
    L2 = "l2"
    MEAN_CENTERED = "mean_centered"
    STANDARDIZED = "standardized"
    WHITENED = "whitened"
    CUSTOM = "custom"


class TransformationMethod(StrEnum):
    MEAN_CENTER = "mean_center"


class MissingAnnotationPolicy(StrEnum):
    ERROR = "error"
    DROP = "drop"


class Metric(StrEnum):
    COSINE_SIMILARITY = "cosine_similarity"
    COSINE_DISTANCE = "cosine_distance"
    ANGULAR_DISTANCE = "angular_distance"
    CORRELATION_DISTANCE = "correlation_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    LENGTH_RATIO = "length_ratio"
    SEQUENCE_IDENTITY = "sequence_identity"
    KMER_JACCARD = "kmer_jaccard"
    BLOSUM62_GLOBAL = "blosum62_global"
    BLOSUM62_LOCAL = "blosum62_local"
    RESIDUE_GLOBAL_ALIGNMENT = "residue_global_alignment"
    RESIDUE_LOCAL_ALIGNMENT = "residue_local_alignment"
    RESIDUE_BEST_MATCH = "residue_best_match"
    RESIDUE_OPTIMAL_TRANSPORT = "residue_optimal_transport"


class Visualization(StrEnum):
    CLUSTERED_HEATMAP = "clustered_heatmap"
    DENDROGRAM = "dendrogram"
    PCOA = "pcoa"
    UMAP = "umap"
    NEAREST_NEIGHBOR_GRAPH = "nearest_neighbor_graph"
    RESIDUE_SIMILARITY_MAP = "residue_similarity_map"
    MODEL_AGREEMENT = "model_agreement"
    DISTANCE_DISTRIBUTION = "distance_distribution"
    MODEL_MATRIX_SMALL_MULTIPLES = "model_matrix_small_multiples"
    CONSENSUS_HEATMAP = "consensus_heatmap"
    MODEL_DISAGREEMENT = "model_disagreement"
    RESIDUE_MODEL_AGREEMENT = "residue_model_agreement"


class CrossModelMethod(StrEnum):
    MATRIX_SPEARMAN = "matrix_spearman"
    MATRIX_PEARSON = "matrix_pearson"
    LINEAR_CKA = "linear_cka"
    NEIGHBOR_OVERLAP = "neighbor_overlap"
    CONSENSUS_RANK = "consensus_rank"
    PAIR_DISAGREEMENT = "pair_disagreement"
    RESIDUE_CKA = "residue_cka"
    RESIDUE_RSA = "residue_rsa"


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    end: int

    def validate(self, length: int) -> None:
        if self.start < 1 or self.end < self.start or self.end > length:
            raise ValidationError(
                f"Region {self.name!r} must use 1-based inclusive coordinates within 1..{length}"
            )


@dataclass(frozen=True)
class ProteinSequence:
    id: str
    sequence: str
    metadata: dict[str, Any] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()


@dataclass(frozen=True)
class RepresentationSpec:
    method: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int = "model_default"
    normalize: Normalization = Normalization.L2
    window_size: int | None = None
    window_step: int | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    precision: str = "auto"
    device: str = "auto"
    all_to_all: str = "exact"
    on_long_sequence: str = "error"
    allow_partial: bool = True
    max_exact_sequences: int = 2_000


@dataclass(frozen=True)
class CrossModelSpec:
    enabled: bool = False
    methods: tuple[CrossModelMethod, ...] = (CrossModelMethod.MATRIX_SPEARMAN,)
    representation: RepresentationMethod = RepresentationMethod.MEAN
    neighbor_k: tuple[int, ...] = (1, 5, 10)
    max_disagreements: int = 20


@dataclass(frozen=True)
class DiagnosticSpec:
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    sample_pairs: int = 10_000
    hubness_k: int = 10
    seed: int = 42
    max_exact_hubness_sequences: int = 2_000


@dataclass(frozen=True)
class TransformationSpec:
    method: TransformationMethod = TransformationMethod.MEAN_CENTER
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    fit_sequence_ids: tuple[str, ...] | None = None
    sample_pairs: int = 10_000
    hubness_k: int = 10
    seed: int = 42


@dataclass(frozen=True)
class AnnotationAnalysisSpec:
    feature: str = ""
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    k: tuple[int, ...] = (1, 5, 10)
    missing_policy: MissingAnnotationPolicy = MissingAnnotationPolicy.ERROR
    max_exact_sequences: int = 2_000


@dataclass(frozen=True)
class EmbeddingRequest:
    models: tuple[str, ...]
    sequences: tuple[ProteinSequence, ...]
    representations: tuple[RepresentationSpec, ...] = (RepresentationSpec(),)
    execution: ExecutionOptions = ExecutionOptions()


@dataclass(frozen=True)
class CompareRequest(EmbeddingRequest):
    metrics: tuple[Metric, ...] = (Metric.COSINE_SIMILARITY,)
    visualizations: tuple[Visualization, ...] = ()
    cross_model: CrossModelSpec = CrossModelSpec()


@dataclass(frozen=True)
class Availability:
    available: bool
    reason_code: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    id: str
    display_name: str
    family: str
    provider: str
    artifact: str
    revision: str
    license: str
    citation: str
    architecture: str
    input_modalities: tuple[str, ...]
    alphabet: str
    ambiguous_residues: str
    tokenizer_convention: str
    max_native_length: int | None
    num_layers: int | None
    embedding_dimension: int | None
    representations: tuple[RepresentationMethod, ...]
    default_layer: str | int
    one_token_per_residue: bool
    special_tokens: str
    devices: tuple[str, ...]
    precisions: tuple[str, ...]
    parameters: int | None
    adapter_version: str = "1"
    conformance: str = "untested"
    limitations: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    backend: str = "huggingface"
    preprocess: str = "raw"
    trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["representations"] = [str(item) for item in self.representations]
        return value


@dataclass
class RawEmbedding:
    sequence_id: str
    residues: list[list[float]]
    native: list[float] | None = None
    token_to_residue: list[int | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepresentationResult:
    sequence_id: str
    method: str
    layer: str | int
    values: Any
    shape: tuple[int, ...]
    norm_before: float | None
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)
