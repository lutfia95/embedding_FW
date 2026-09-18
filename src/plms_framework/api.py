from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

try:
    from fastapi import FastAPI, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError as exc:  # pragma: no cover - exercised by CLI without API extra
    raise RuntimeError(
        "Install the API dependencies with: pip install 'plms-framework[api]'"
    ) from exc

from . import __version__
from .annotations import analyze_categorical_annotations
from .artifacts import ArtifactStore
from .cross_model import compare_cross_model
from .diagnostics import diagnose_embeddings
from .errors import FrameworkError, ValidationError
from .external import build_imported_space
from .jobs import JobManager
from .models import (
    AnnotationAnalysisSpec,
    CrossModelMethod,
    CrossModelSpec,
    DiagnosticSpec,
    ImportedNormalization,
    Metric,
    MissingAnnotationPolicy,
    Normalization,
    RepresentationMethod,
    TransformationMethod,
    TransformationSpec,
    Visualization,
)
from .pipeline import Pipeline
from .registry import default_registry
from .sequences import parse_fasta
from .serialization import request_from_dict
from .transformations import transform_embeddings
from .visualization import render_requested


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegionInput(StrictModel):
    name: str
    start: int = Field(ge=1)
    end: int = Field(ge=1)


class SequenceInput(StrictModel):
    id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    regions: list[RegionInput] = Field(default_factory=list)


class RepresentationInput(StrictModel):
    method: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int = "model_default"
    normalize: Normalization = Normalization.L2
    window_size: int | None = Field(default=None, gt=0)
    window_step: int | None = Field(default=None, gt=0)


class ExecutionInput(StrictModel):
    precision: Literal["auto", "float32", "float16", "fp16", "bfloat16", "bf16"] = (
        "auto"
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    all_to_all: Literal["exact"] = "exact"
    on_long_sequence: Literal["error"] = "error"
    allow_partial: bool = True
    max_exact_sequences: int = Field(default=2_000, ge=2)


class EmbedInput(StrictModel):
    models: list[str] = Field(min_length=1)
    sequences: list[SequenceInput] = Field(min_length=1)
    representations: list[RepresentationInput] = Field(
        default_factory=lambda: [RepresentationInput()]
    )
    execution: ExecutionInput = Field(default_factory=ExecutionInput)


class CrossModelInput(StrictModel):
    enabled: bool = False
    methods: list[CrossModelMethod] = Field(
        default_factory=lambda: [CrossModelMethod.MATRIX_SPEARMAN]
    )
    representation: RepresentationMethod = RepresentationMethod.MEAN
    neighbor_k: list[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [1, 5, 10])
    max_disagreements: int = Field(default=20, gt=0)


class CompareInput(EmbedInput):
    metrics: list[Metric] = Field(default_factory=lambda: [Metric.COSINE_SIMILARITY])
    visualizations: list[Visualization] = Field(default_factory=list)
    cross_model: CrossModelInput = Field(default_factory=CrossModelInput)


class CrossCompareSource(StrictModel):
    job_id: str


class CrossCompareInput(StrictModel):
    sources: list[CrossCompareSource] = Field(min_length=1)
    methods: list[CrossModelMethod] = Field(
        default_factory=lambda: [CrossModelMethod.MATRIX_SPEARMAN]
    )
    representation: RepresentationMethod = RepresentationMethod.MEAN
    neighbor_k: list[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [1, 5, 10])
    max_disagreements: int = Field(default=20, gt=0)
    visualizations: list[Visualization] = Field(default_factory=list)


class ImportedEmbeddingInput(StrictModel):
    sequence_id: str = Field(min_length=1, max_length=256)
    values: list[float] | list[list[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("values", mode="before")
    @classmethod
    def reject_coerced_embedding_values(cls, value):
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, bool) or not isinstance(item, (int, float)):
                # Pydantic field validators translate ValueError into a 422 response.
                raise ValueError("embedding values must be JSON numbers")  # noqa: TRY004
        return value


class ImportedRepresentationInput(StrictModel):
    method: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int = "external"
    normalization: ImportedNormalization = ImportedNormalization.UNKNOWN
    embeddings: list[ImportedEmbeddingInput] = Field(min_length=1)


class ImportedSourceInput(StrictModel):
    display_name: str | None = None
    family: str | None = None
    provider: str | None = None
    artifact: str | None = None
    revision: str | None = None
    license: str | None = None
    citation: str | None = None
    architecture: str | None = None
    tokenizer_convention: str | None = None
    one_token_per_residue: bool | None = None
    special_tokens: str | None = None
    parameters: int | None = Field(default=None, ge=0)
    preprocess: str | None = None
    trust_remote_code: bool | None = None


class ImportSpaceInput(StrictModel):
    space_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    representations: list[ImportedRepresentationInput] = Field(min_length=1)
    source: ImportedSourceInput = Field(default_factory=ImportedSourceInput)


class DiagnosticsInput(StrictModel):
    source_job_id: str = Field(min_length=1)
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    sample_pairs: int = Field(default=10_000, ge=1, le=100_000)
    hubness_k: int = Field(default=10, ge=1)
    seed: int = 42


class TransformationInput(StrictModel):
    source_job_id: str = Field(min_length=1)
    method: TransformationMethod = TransformationMethod.MEAN_CENTER
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    fit_sequence_ids: list[str] | None = None
    sample_pairs: int = Field(default=10_000, ge=1, le=100_000)
    hubness_k: int = Field(default=10, ge=1)
    seed: int = 42


class AnnotationRecordInput(StrictModel):
    sequence_id: str = Field(min_length=1, max_length=256)
    categorical: dict[str, str | int | bool | None] = Field(default_factory=dict)
    numerical: dict[str, float | None] = Field(default_factory=dict)


class AnnotationAnalysisInput(StrictModel):
    source_job_id: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    annotations: list[AnnotationRecordInput] = Field(min_length=1)
    representation: RepresentationMethod = RepresentationMethod.MEAN
    layer: str | int | None = None
    k: list[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [1, 5, 10])
    missing_policy: MissingAnnotationPolicy = MissingAnnotationPolicy.ERROR


class BenchmarkInput(StrictModel):
    request: EmbedInput
    track: str = "efficiency"
    repetitions: int = Field(default=1, ge=1, le=20)


artifact_root = Path(os.environ.get("PLMS_ARTIFACT_DIR", ".plms_artifacts"))
store = ArtifactStore(artifact_root)
jobs = JobManager(store, max_workers=int(os.environ.get("PLMS_JOB_WORKERS", "1")))
pipeline = Pipeline(default_registry)

app = FastAPI(
    title="PLMs Framework API",
    version=__version__,
    description="Capability-driven protein language model embeddings and comparisons.",
)


@app.exception_handler(FrameworkError)
async def framework_error_handler(_: Request, exc: FrameworkError):
    return JSONResponse(status_code=422, content=exc.as_dict())


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    available = sum(adapter.availability().available for _, adapter in default_registry.models())
    return {
        "status": "ready",
        "registered_models": sum(1 for _ in default_registry.models()),
        "available_models": available,
    }


@app.get("/v1/models")
def list_models(available_only: bool = False, family: str | None = None) -> dict:
    models = []
    for manifest, adapter in default_registry.models():
        availability = adapter.availability()
        if available_only and not availability.available:
            continue
        if family and manifest.family.lower() != family.lower():
            continue
        models.append({**manifest.to_dict(), "availability": availability.__dict__})
    return {"models": models, "count": len(models)}


@app.get("/v1/models/{model_id}")
def get_model(model_id: str) -> dict:
    adapter = default_registry.adapter(model_id)
    return {**adapter.manifest.to_dict(), "availability": adapter.availability().__dict__}


@app.post("/v1/estimate")
def estimate(body: EmbedInput) -> dict:
    return pipeline.estimate(request_from_dict(body.model_dump(), compare=False))


@app.post("/v1/embed")
def embed(body: EmbedInput) -> dict:
    return pipeline.embed(request_from_dict(body.model_dump(), compare=False))


@app.post("/v1/spaces/import", status_code=202)
def import_space(body: ImportSpaceInput) -> dict:
    payload = body.model_dump(mode="json")
    return jobs.submit(
        "import_space",
        lambda: build_imported_space(
            space_name=payload["space_name"],
            representations=payload["representations"],
            source={key: value for key, value in payload["source"].items() if value is not None},
        ),
    ).public()


@app.post("/v1/diagnostics", status_code=202)
def diagnostics(body: DiagnosticsInput) -> dict:
    source_job = jobs.get(body.source_job_id)
    if source_job.status != "succeeded" or source_job.result is None:
        raise ValidationError(
            f"Source job {body.source_job_id} is not a completed successful job",
            context={"job_id": body.source_job_id, "status": source_job.status},
        )
    source_result = source_job.result
    if not source_result.get("models"):
        raise ValidationError(
            "Source job does not contain model representations",
            context={"job_id": body.source_job_id},
        )
    spec = DiagnosticSpec(
        representation=body.representation,
        layer=body.layer,
        sample_pairs=body.sample_pairs,
        hubness_k=body.hubness_k,
        seed=body.seed,
    )

    def run() -> dict:
        diagnostic_result = diagnose_embeddings(
            source_result["models"], source_result.get("sequence_order", []), spec
        )
        return {
            "schema_version": "1.0",
            "source_job": body.source_job_id,
            "sequence_order": source_result.get("sequence_order", []),
            "diagnostics": diagnostic_result,
            "requested_visualizations": [],
            "provenance": {
                "framework_version": __version__,
                "operation": "embedding_diagnostics",
                "source_job": body.source_job_id,
                "diagnostic_spec": asdict(spec),
                "source_provenance": source_result.get("provenance", {}),
            },
        }

    return jobs.submit("diagnostics", run).public()


@app.post("/v1/transform", status_code=202)
def transform(body: TransformationInput) -> dict:
    source_job = jobs.get(body.source_job_id)
    if source_job.status != "succeeded" or source_job.result is None:
        raise ValidationError(
            f"Source job {body.source_job_id} is not a completed successful job",
            context={"job_id": body.source_job_id, "status": source_job.status},
        )
    source_result = source_job.result
    if not source_result.get("models"):
        raise ValidationError(
            "Source job does not contain model representations",
            context={"job_id": body.source_job_id},
        )
    spec = TransformationSpec(
        method=body.method,
        representation=body.representation,
        layer=body.layer,
        fit_sequence_ids=tuple(body.fit_sequence_ids)
        if body.fit_sequence_ids is not None
        else None,
        sample_pairs=body.sample_pairs,
        hubness_k=body.hubness_k,
        seed=body.seed,
    )

    def run() -> dict:
        result = transform_embeddings(
            source_result["models"], source_result.get("sequence_order", []), spec
        )
        result["source_job"] = body.source_job_id
        result["provenance"].update(
            {
                "framework_version": __version__,
                "source_job": body.source_job_id,
                "source_provenance": source_result.get("provenance", {}),
            }
        )
        return result

    return jobs.submit("transformation", run, render=_render_transformation).public()


def _render_transformation(result: dict, artifact_store: ArtifactStore, job_id: str) -> list[dict]:
    return [
        artifact_store.save_json(
            job_id,
            "transformation_parameters.json",
            result.get("transformation_parameters", {}),
        )
    ]


@app.post("/v1/annotations/analyze", status_code=202)
def analyze_annotations(body: AnnotationAnalysisInput) -> dict:
    source_job = jobs.get(body.source_job_id)
    if source_job.status != "succeeded" or source_job.result is None:
        raise ValidationError(
            f"Source job {body.source_job_id} is not a completed successful job",
            context={"job_id": body.source_job_id, "status": source_job.status},
        )
    source_result = source_job.result
    if not source_result.get("models"):
        raise ValidationError(
            "Source job does not contain model representations",
            context={"job_id": body.source_job_id},
        )
    spec = AnnotationAnalysisSpec(
        feature=body.feature,
        representation=body.representation,
        layer=body.layer,
        k=tuple(body.k),
        missing_policy=body.missing_policy,
    )
    annotations = [item.model_dump(mode="json") for item in body.annotations]

    def run() -> dict:
        analysis = analyze_categorical_annotations(
            source_result["models"],
            source_result.get("sequence_order", []),
            annotations,
            spec,
        )
        return {
            "schema_version": "1.0",
            "source_job": body.source_job_id,
            "sequence_order": source_result.get("sequence_order", []),
            "annotation_analysis": analysis,
            "requested_visualizations": [],
            "provenance": {
                "framework_version": __version__,
                "operation": "categorical_annotation_analysis",
                "source_job": body.source_job_id,
                "annotation_spec": asdict(spec),
                "annotations_sha256": analysis["annotations_sha256"],
                "source_provenance": source_result.get("provenance", {}),
            },
        }

    return jobs.submit("annotation_analysis", run).public()


@app.post("/v1/compare", status_code=202)
def compare(body: CompareInput) -> dict:
    domain = request_from_dict(body.model_dump(), compare=True)
    job = jobs.submit("compare", lambda: pipeline.compare(domain), render=render_requested)
    return job.public()


@app.post("/v1/cross-compare", status_code=202)
def cross_compare(body: CrossCompareInput) -> dict:
    source_results = []
    for source in body.sources:
        source_job = jobs.get(source.job_id)
        if source_job.status != "succeeded" or source_job.result is None:
            raise ValidationError(
                f"Source job {source.job_id} is not a completed successful job",
                context={"job_id": source.job_id, "status": source_job.status},
            )
        source_results.append((source.job_id, source_job.result))
    try:
        spec = CrossModelSpec(
            enabled=True,
            methods=tuple(CrossModelMethod(item) for item in body.methods),
            representation=RepresentationMethod(body.representation),
            neighbor_k=tuple(body.neighbor_k),
            max_disagreements=body.max_disagreements,
        )
        requested_visualizations = [str(Visualization(item)) for item in body.visualizations]
    except ValueError as exc:
        raise ValidationError(f"Invalid cross-model request: {exc}") from exc

    def run() -> dict:
        sequence_order = source_results[0][1].get("sequence_order", [])
        merged_models = {}
        for source_job_id, result in source_results:
            if result.get("sequence_order", []) != sequence_order:
                raise ValidationError(
                    "Cross-compare source jobs must use identical sequence IDs and ordering",
                    context={"job_id": source_job_id},
                )
            for model_id, output in result.get("models", {}).items():
                if model_id in merged_models:
                    raise ValidationError(
                        f"Model {model_id} occurs in more than one source job",
                        context={"model": model_id},
                    )
                merged_models[model_id] = output
        cross_result = compare_cross_model(merged_models, sequence_order, spec)
        return {
            "schema_version": "1.0",
            "source_jobs": [job_id for job_id, _ in source_results],
            "sequence_order": sequence_order,
            "models": {
                model_id: {
                    "manifest": output.get("manifest", {}),
                    "matrices": output.get("matrices", {}),
                }
                for model_id, output in merged_models.items()
            },
            "cross_model": cross_result,
            "model_agreement": _legacy_model_agreement(cross_result),
            "requested_visualizations": requested_visualizations,
            "provenance": {
                "framework_version": __version__,
                "operation": "cross_compare_existing_jobs",
                "cross_model": asdict(spec),
                "source_jobs": [
                    {
                        "job_id": job_id,
                        "provenance": result.get("provenance", {}),
                    }
                    for job_id, result in source_results
                ],
            },
        }

    return jobs.submit("cross_compare", run, render=render_requested).public()


def _legacy_model_agreement(cross_result: dict) -> dict:
    spearman = cross_result.get("comparisons", {}).get("matrix_spearman", {})
    if not spearman.get("available"):
        return {"model_order": [], "spearman": []}
    return {
        "model_order": spearman.get("model_order", []),
        "spearman": spearman.get("values", []),
    }


@app.post("/v1/benchmark", status_code=202)
def benchmark(body: BenchmarkInput) -> dict:
    request = request_from_dict(body.request.model_dump(), compare=False)

    def run() -> dict:
        if body.track != "efficiency":
            return {
                "track": body.track,
                "status": "not_configured",
                "message": "Biological tracks require a versioned benchmark dataset manifest",
            }
        timings = []
        last = None
        for _ in range(body.repetitions):
            started = time.perf_counter()
            last = pipeline.embed(request)
            timings.append(time.perf_counter() - started)
        return {
            "track": "efficiency",
            "repetitions": body.repetitions,
            "seconds": timings,
            "mean_seconds": sum(timings) / len(timings),
            "estimate": pipeline.estimate(request),
            "last_run_provenance": last.get("provenance", {}) if last else {},
        }

    return jobs.submit("benchmark", run).public()


@app.post("/v1/fasta/parse")
async def fasta_parse(file: UploadFile) -> dict:
    data = await file.read()
    records = parse_fasta(data.decode("utf-8"))
    return {"sequences": [{"id": item.id, "sequence": item.sequence} for item in records]}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, include_result: bool = False) -> dict:
    return jobs.get(job_id).public(include_result=include_result)


@app.get("/v1/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str) -> dict:
    jobs.get(job_id)
    return {"artifacts": store.list(job_id)}


@app.get("/v1/jobs/{job_id}/artifacts/{name}")
def download_artifact(job_id: str, name: str):
    jobs.get(job_id)
    path = store.path(job_id, name)
    return FileResponse(path)
