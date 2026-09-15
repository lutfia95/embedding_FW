# Using PLMs Framework from Python

PLMs Framework can be used as a normal Python library. You do not need to run
the FastAPI server when the embedding and comparison are part of a notebook,
analysis script, training pipeline, or another Python application.

This guide documents the currently implemented Python interface, every request
option accepted by the core pipeline, the result structure, and direct Python
access to diagnostics, transformations, annotation analysis, imported
embeddings, FASTA parsing, and visualization rendering.

## Contents

- [Installation](#installation)
- [Choose between Python and the API](#choose-between-python-and-the-api)
- [Quick start](#quick-start)
- [Dictionary requests and typed requests](#dictionary-requests-and-typed-requests)
- [Discovering models](#discovering-models)
- [Sequence options](#sequence-options)
- [Representation options](#representation-options)
- [Execution options](#execution-options)
- [Embedding](#embedding)
- [Within-model comparison](#within-model-comparison)
- [Cross-PLM comparison](#cross-plm-comparison)
- [FASTA input](#fasta-input)
- [Understanding results](#understanding-results)
- [Rendering visualizations](#rendering-visualizations)
- [Importing existing embeddings](#importing-existing-embeddings)
- [Embedding diagnostics](#embedding-diagnostics)
- [Mean-centering](#mean-centering)
- [Categorical annotation analysis](#categorical-annotation-analysis)
- [Errors and partial failures](#errors-and-partial-failures)
- [Reproducibility and resource guidance](#reproducibility-and-resource-guidance)

## Installation

Python 3.11 or newer is required. From the repository root, install the package
in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core installation has no mandatory third-party dependencies and includes
the deterministic reference adapter. Install the Hugging Face dependencies to
run compatible real checkpoints:

```bash
python -m pip install -e '.[hf]'
```

Install advanced interactive visualizations with:

```bash
python -m pip install -e '.[visual]'
```

The API dependencies are not needed for direct Python use. Install all runtime
features only when they are required:

```bash
python -m pip install -e '.[all]'
```

For development without installation, commands may instead set
`PYTHONPATH=src`, but an editable install is more convenient for notebooks and
applications.

## Choose between Python and the API

The direct Python interface and FastAPI service use the same model registry,
adapters, representations, and comparison algorithms. Their execution models
differ:

| Capability | Direct Python | FastAPI service |
|---|---|---|
| Embed sequences | `Pipeline.embed()` | `POST /v1/embed` |
| Estimate resources | `Pipeline.estimate()` | `POST /v1/estimate` |
| Compare sequences and PLMs | `Pipeline.compare()` | `POST /v1/compare` |
| Execution | Runs in the calling process | Comparison jobs run in a worker |
| Result | Returned as a Python `dict` | Returned as JSON/job result |
| Job persistence and polling | Not automatic | Built in |
| Artifact rendering | Explicit function call | Automatic for comparison jobs |

Use direct Python when the caller already owns the execution process and wants
the numerical result immediately. Use the API when clients need HTTP access,
background jobs, polling, and managed artifacts.

## Quick start

The built-in `reference` model is deterministic, dependency-free, and useful
for learning the interface. It is not a biological PLM.

```python
from plms_framework import Pipeline
from plms_framework.serialization import request_from_dict

request = request_from_dict(
    {
        "models": ["reference"],
        "sequences": [
            {"id": "protein_a", "sequence": "MKTAYIAK"},
            {"id": "protein_b", "sequence": "MKTAYVAK"},
            {"id": "protein_c", "sequence": "GGGGGGGG"},
        ],
        "representations": [
            {"method": "mean", "normalize": "l2"},
        ],
        "metrics": [
            "cosine_similarity",
            "euclidean_distance",
            "sequence_identity",
        ],
    },
    compare=True,
)

result = Pipeline().compare(request)

model_id = "reference-deterministic-v1"
similarity = result["models"][model_id]["matrices"][
    "mean@model_default:cosine_similarity"
]

print(result["sequence_order"])
print(similarity)
print(result["sequence_baselines"]["sequence_identity"])
```

Aliases are accepted as input. Results use the canonical model ID, which is why
the output key above is `reference-deterministic-v1` rather than `reference`.

To use the tested ESM-2 8M checkpoint, install the `hf` extra and change the
model list:

```python
request_data = {
    "models": ["esm2-8m"],
    "sequences": [{"id": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"}],
    "representations": [
        {"method": "mean", "normalize": "l2"},
        {"method": "residue", "normalize": "none"},
    ],
}

request = request_from_dict(request_data, compare=False)
result = Pipeline().embed(request)
```

The first run may download the checkpoint. Checkpoint access, local storage,
memory, and upstream license conditions remain the caller's responsibility.

## Dictionary requests and typed requests

There are two equivalent ways to construct a request.

### Dictionary construction

`request_from_dict()` is usually easiest for configuration files, notebooks,
and programmatically assembled requests:

```python
from plms_framework.serialization import request_from_dict

embed_request = request_from_dict(data, compare=False)
compare_request = request_from_dict(data, compare=True)
```

- `compare=False` creates an `EmbeddingRequest` for `embed()` or `estimate()`.
- `compare=True` creates a `CompareRequest` for `compare()`.
- Unknown enum values and malformed required fields become a
  `ValidationError`.

Unlike the FastAPI Pydantic schema, dictionary parsing does not reject every
unknown extra key. Treat the documented fields as the supported contract.

### Fully typed construction

Typed dataclasses and enums are useful in application code and provide editor
completion:

```python
from plms_framework import (
    CompareRequest,
    CrossModelMethod,
    CrossModelSpec,
    ExecutionOptions,
    Pipeline,
    ProteinSequence,
    RepresentationSpec,
)
from plms_framework.models import (
    Metric,
    Normalization,
    RepresentationMethod,
    Visualization,
)

request = CompareRequest(
    models=("reference",),
    sequences=(
        ProteinSequence(id="a", sequence="MKTAYIAK"),
        ProteinSequence(id="b", sequence="MKTAYVAK"),
    ),
    representations=(
        RepresentationSpec(
            method=RepresentationMethod.MEAN,
            layer="model_default",
            normalize=Normalization.L2,
        ),
    ),
    metrics=(Metric.COSINE_SIMILARITY, Metric.SEQUENCE_IDENTITY),
    visualizations=(Visualization.CLUSTERED_HEATMAP,),
    cross_model=CrossModelSpec(
        enabled=False,
        methods=(CrossModelMethod.MATRIX_SPEARMAN,),
    ),
    execution=ExecutionOptions(allow_partial=True),
)

result = Pipeline().compare(request)
```

Use tuples for tuple-typed fields in typed requests. The dictionary parser
converts JSON-style lists to the required tuples automatically.

## Discovering models

Model discovery does not load or download checkpoints:

```python
from dataclasses import asdict

from plms_framework import default_registry

for manifest, adapter in default_registry.models():
    print(
        manifest.id,
        manifest.family,
        manifest.embedding_dimension,
        asdict(adapter.availability()),
    )
```

Inspect one manifest or resolve an alias:

```python
manifest = default_registry.manifest("esm2-8m")
canonical_id = default_registry.resolve_id("reference")

print(manifest.to_dict())
print(canonical_id)  # reference-deterministic-v1
```

The currently registered IDs are:

| Model ID | Family | Dimension | Conformance status |
|---|---|---:|---|
| `reference-deterministic-v1` | Reference fixture | 16 | Tested; not a biological PLM |
| `esm2-8m` | ESM-2 | 320 | Tested for CPU shapes, masking, and batch invariance |
| `esm2-35m` | ESM-2 | 480 | Adapter registered; checkpoint validation required |
| `esm2-150m` | ESM-2 | 640 | Adapter registered; checkpoint validation required |
| `esm2-650m` | ESM-2 | 1,280 | Adapter registered; checkpoint validation required |
| `esm2-3b` | ESM-2 | 2,560 | Adapter registered; checkpoint validation required |
| `esm2-15b` | ESM-2 | 5,120 | Adapter registered; checkpoint validation required |
| `esmc-300m` | ESM-C | Not declared | Generic adapter entry; access, compatibility, and validation required |
| `esmc-600m` | ESM-C | Not declared | Generic adapter entry; access, compatibility, and validation required |
| `esmc-6b` | ESM-C | Not declared | Generic adapter entry; access, compatibility, and validation required |
| `prott5-xl-uniref50` | ProtTrans | 1,024 | Adapter registered; checkpoint validation required |
| `protbert-bfd` | ProtTrans | 1,024 | Adapter registered; checkpoint validation required |
| `ankh-base` | Ankh | 768 | Adapter registered; checkpoint validation required |
| `ankh-large` | Ankh | 1,536 | Adapter registered; checkpoint validation required |

Model aliases include `reference`, upstream ESM-2 checkpoint names, and the
ProtTrans underscore-form artifact names recorded in each manifest.

`adapter.availability().available` currently means that the adapter's required
Python packages are importable. It does not prove that a checkpoint is cached,
downloadable, licensed for the intended use, compatible with the adapter, or
small enough for the selected hardware. The manifest's `conformance` field and
`limitations` list provide the stronger qualification.

The ESM-C entries currently use the generic Hugging Face adapter path and have
not passed model-level conformance testing. Treat them as discoverable registry
entries, not as guaranteed runnable integrations.

## Sequence options

Each sequence has the following fields:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `id` | `str` | Yes | Unique identifier retained in result ordering |
| `sequence` | `str` | Yes | Amino-acid sequence |
| `metadata` | `dict` | No | Caller-owned metadata; default `{}` |
| `regions` | list/tuple of regions | No | Named ranges used by `region_mean` |

Example with metadata and regions:

```python
from plms_framework.serialization import request_from_dict

request = request_from_dict(
    {
        "models": ["reference"],
        "sequences": [
            {
                "id": "kinase_1",
                "sequence": "MKTAYIAKQRQISFVKSHFSRQ",
                "metadata": {"family": "example", "organism": "synthetic"},
                "regions": [
                    {"name": "n_terminal", "start": 1, "end": 8},
                    {"name": "c_terminal", "start": 15, "end": 22},
                ],
            }
        ],
        "representations": [{"method": "region_mean"}],
    },
    compare=False,
)
```

Sequence handling rules:

- IDs are stripped and must be non-empty and unique after stripping.
- Sequences are uppercased and all formatting whitespace is removed.
- Empty sequences are rejected.
- Regions use 1-based inclusive coordinates.
- A region must satisfy `1 <= start <= end <= sequence length`.
- The default alphabet is `ACDEFGHIKLMNPQRSTVWYBXZJUO`.
- Unsupported characters are rejected rather than silently replaced.
- Long sequences are rejected if they exceed a model's declared native limit.
- Silent truncation is never performed.

Sequence metadata is accepted and preserved while the request is processed,
but the current numerical result does not copy the complete metadata mapping
into every output record. Keep the original `id` to metadata mapping alongside
the returned result when downstream analysis needs it.

## Representation options

A representation controls how residue-level model output becomes a result for
each protein.

| Method | Output per sequence | Use |
|---|---|---|
| `mean` | Vector of shape `D` | Portable default for protein-level comparison |
| `max` | Vector of shape `D` | Feature-wise maximum; exploratory diagnostic |
| `residue` | Matrix of shape `L x D` | Residue comparison, motifs, alignments, local analysis |
| `region_mean` | `R` named vectors of shape `D` | User-defined domains or sequence ranges |
| `window_mean` | `W` positional vectors of shape `D` | Deterministic local windows |
| `native` | Vector of shape `D` | Upstream-defined pooled output when an adapter explicitly supports it |

All currently registered built-in and Hugging Face manifests advertise
`mean`, `max`, `residue`, `region_mean`, and `window_mean`. `native` exists in
the data model and is accepted for imported spaces, but no current default
registry manifest advertises it. Requesting `native` from those models produces
an unsupported-capability failure.

Every representation specification accepts:

| Option | Supported values | Default | Notes |
|---|---|---|---|
| `method` | Table above | `mean` | Pooling or residue output |
| `layer` | `"model_default"` or integer | `"model_default"` | Integer range is checked when layer count is known |
| `normalize` | `"l2"`, `"none"` | `"l2"` | Applied after pooling, or row-by-row for `residue` |
| `window_size` | Positive integer or `None` | `None` | Required for `window_mean` |
| `window_step` | Positive integer or `None` | `None` | Defaults to `window_size` |

Mean pooling excludes padding and special boundary tokens because adapters must
first return exactly one row for each native residue. L2 normalization retains
the original norm in `norm_before`; residue normalization retains all original
row norms in metadata.

### Request several representations

```python
representations = [
    {"method": "mean", "normalize": "l2"},
    {"method": "max", "normalize": "l2"},
    {"method": "residue", "normalize": "none"},
    {"method": "window_mean", "window_size": 9, "window_step": 4},
]
```

`window_mean` retains the final partial window. `region_mean` requires at least
one region on every sequence being materialized.

### Layer selection

Use `"model_default"` unless the analysis specifically concerns intermediate
layers. For adapters with a known layer count, integer layers range from zero
through the manifest's layer count. An integer causes compatible Hugging Face
models to request hidden states, which uses more memory.

Do not compare matrices built from different layers as if they were the same
representation. The matrix key records both method and layer.

## Execution options

`execution` controls inference and safety behavior:

| Option | Supported/current values | Default | Behavior |
|---|---|---|---|
| `device` | `auto`, `cpu`, `cuda`, `mps` | `auto` | `auto` chooses CUDA, then MPS, then CPU in the Hugging Face adapter |
| `precision` | `auto`, `float32`, `float16`/`fp16`, `bfloat16`/`bf16` | `auto` | Half/bfloat dtypes are passed at model load; compatibility depends on hardware/model |
| `all_to_all` | `exact` | `exact` | Approximate comparison is not implemented |
| `on_long_sequence` | `error` | `error` | Windowed model inference is not implemented |
| `allow_partial` | Boolean | `True` | Isolate model failures instead of aborting a multi-model run |
| `max_exact_sequences` | Integer at least 2 | `2000` | Maximum sequence count accepted by exact comparison |

The dataclass layer stores `device`, `precision`, `all_to_all`, and
`on_long_sequence` as strings. The values listed above are the implemented
ones. In particular, changing `all_to_all` or `on_long_sequence` to a future
mode does not activate an approximate or chunked algorithm.

Example:

```python
"execution": {
    "device": "cuda",
    "precision": "float16",
    "all_to_all": "exact",
    "on_long_sequence": "error",
    "allow_partial": True,
    "max_exact_sequences": 1000,
}
```

## Embedding

Use `Pipeline.embed()` for one or more sequences when pairwise matrices are not
needed:

```python
from plms_framework import Pipeline
from plms_framework.serialization import request_from_dict

request = request_from_dict(
    {
        "models": ["reference"],
        "sequences": [
            {"id": "a", "sequence": "MKTAYIAK"},
            {"id": "b", "sequence": "GGGGGGGG"},
        ],
        "representations": [
            {"method": "mean"},
            {"method": "residue", "normalize": "none"},
        ],
    },
    compare=False,
)

pipeline = Pipeline()
estimate = pipeline.estimate(request)
result = pipeline.embed(request)

print(estimate["models"])
print(result["models"].keys())
```

`estimate()` reports residues, parameter counts, estimated float32 embedding
bytes, distinct pair count, and float64 matrix bytes per metric. It is a
planning estimate, not a guarantee of peak inference memory.

### Extract one vector

Representations are returned as records so several methods and layers can
coexist:

```python
model_output = result["models"]["reference-deterministic-v1"]

mean_a = next(
    item["values"]
    for item in model_output["representations"]
    if item["sequence_id"] == "a" and item["method"] == "mean"
)
```

## Within-model comparison

Use `Pipeline.compare()` with at least two sequences. It embeds the sequences
and computes all requested compatible metrics independently inside each model.

### Supported vector metrics

Vector metrics apply to `mean`, `max`, or supported `native` vectors:

| Metric | Direction | Definition/use |
|---|---|---|
| `cosine_similarity` | Higher is closer | Scale-invariant angular similarity |
| `cosine_distance` | Lower is closer | `1 - cosine_similarity` |
| `angular_distance` | Lower is closer | `acos(cosine) / pi` |
| `correlation_distance` | Lower is closer | Cosine distance after centering each vector's features |
| `euclidean_distance` | Lower is closer | Scale-sensitive L2 distance |

For L2-normalized vectors, cosine and Euclidean distance encode the same pairwise
ordering. Request both only when their different numerical scales help a
downstream consumer.

### Supported sequence metrics

Sequence metrics use the amino-acid strings and are reported once under
`sequence_baselines`, not repeated for each model:

| Metric | Direction | Current definition |
|---|---|---|
| `length_ratio` | Higher is closer | Shorter length divided by longer length |
| `sequence_identity` | Higher is closer | Global Needleman-Wunsch identity |
| `kmer_jaccard` | Higher is closer | Jaccard similarity of unique 3-mers |
| `blosum62_global` | Higher is closer | Normalized global BLOSUM62 score |
| `blosum62_local` | Higher is closer | Normalized local BLOSUM62 score |

### Supported residue metrics

These require `{"method": "residue"}` in the same request:

| Metric | Result |
|---|---|
| `residue_global_alignment` | Global alignment over residue cosine scores |
| `residue_local_alignment` | Local alignment over residue cosine scores |
| `residue_best_match` | Mean bidirectional best-residue similarity |
| `residue_optimal_transport` | Balanced entropy-regularized Sinkhorn similarity |

Residue work can be much larger than pooled-vector comparison. Comparing
proteins of lengths `L1` and `L2` constructs an `L1 x L2` residue-similarity
matrix for that pair.

### Complete comparison example

```python
request = request_from_dict(
    {
        "models": ["reference"],
        "sequences": [
            {"id": "a", "sequence": "MKTAYIAK"},
            {"id": "b", "sequence": "MKTAYVAK"},
            {"id": "c", "sequence": "GGGGGGGG"},
        ],
        "representations": [
            {"method": "mean", "normalize": "l2"},
            {"method": "residue", "normalize": "none"},
        ],
        "metrics": [
            "cosine_similarity",
            "angular_distance",
            "sequence_identity",
            "blosum62_global",
            "residue_local_alignment",
            "residue_best_match",
        ],
        "execution": {"max_exact_sequences": 2000},
    },
    compare=True,
)

result = Pipeline().compare(request)

model_output = result["models"]["reference-deterministic-v1"]
print(model_output["matrices"].keys())
print(model_output["residue_pairs"]["a::b"])
```

Matrix keys have the form:

```text
METHOD@LAYER:METRIC
```

For example, `mean@model_default:cosine_similarity`.

## Cross-PLM comparison

Raw embedding coordinates from different PLMs must not be compared directly.
Their dimensions and coordinate systems are unrelated. The framework instead
compares the relationship structures each PLM learns over shared observations.

### Multiple sequences across models

Use at least two models and request the pooled representation named in
`cross_model.representation`:

```python
request = request_from_dict(
    {
        "models": ["esm2-8m", "esm2-35m"],
        "sequences": [
            {"id": "a", "sequence": "MKTAYIAK"},
            {"id": "b", "sequence": "MKTAYVAK"},
            {"id": "c", "sequence": "GGGGGGGG"},
        ],
        "representations": [{"method": "mean", "normalize": "l2"}],
        "metrics": ["cosine_similarity"],
        "cross_model": {
            "enabled": True,
            "representation": "mean",
            "methods": [
                "matrix_spearman",
                "matrix_pearson",
                "linear_cka",
                "neighbor_overlap",
                "consensus_rank",
                "pair_disagreement",
            ],
            "neighbor_k": [1, 2],
            "max_disagreements": 20,
        },
        "execution": {"allow_partial": True},
    },
    compare=True,
)

result = Pipeline().compare(request)
cross = result["cross_model"]
print(cross["scientific_guardrail"])
print(cross["comparisons"]["matrix_spearman"])
```

For a dependency-free demonstration, construct a registry with two reference
adapters under different IDs, as shown later in
[Custom registries](#custom-registries).

Supported multi-sequence methods:

| Method | What is compared | Minimum observations |
|---|---|---:|
| `matrix_spearman` | Ranks of within-model pairwise cosine relationships | 3 sequences |
| `matrix_pearson` | Values of within-model pairwise cosine relationships | 3 sequences |
| `linear_cka` | Centered `N x D` representation matrices | 2 sequences; more are strongly preferable |
| `neighbor_overlap` | Jaccard overlap of top-k neighbors | 2 sequences |
| `consensus_rank` | Mean within-model pair-rank percentile | 3 sequences |
| `pair_disagreement` | Range of within-model pair ranks | 3 sequences |

`neighbor_k` contains positive integers. Values larger than `N - 1` are reduced
to `N - 1`. `max_disagreements` limits the number of most-disagreeing sequence
pairs returned and must be positive.

Matrix correlations need at least three sequences. For interpretable research,
20-50 or more diverse sequences are preferable. Linear CKA with two sequences
is mathematically available but receives an explicit warning because it is
usually uninformative.

### One sequence across models

A single pooled vector from each model cannot support valid cross-model
comparison. Use the shared residues as observations:

```python
request = request_from_dict(
    {
        "models": ["esm2-8m", "esm2-35m"],
        "sequences": [
            {"id": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"},
        ],
        "representations": [
            {"method": "residue", "normalize": "none"},
        ],
        "metrics": [],
        "cross_model": {
            "enabled": True,
            "methods": ["residue_cka", "residue_rsa"],
        },
    },
    compare=True,
)

result = Pipeline().compare(request)
agreement = result["cross_model"]["comparisons"]
```

| Method | Meaning | Minimum sequence length |
|---|---|---:|
| `residue_cka` | Linear CKA with aligned residues as observations | 2 residues |
| `residue_rsa` | Spearman correlation between residue self-similarity matrices | 3 residues |

Both methods require consistent one-token-per-residue mappings. Feature
dimensions may differ between models.

### Cross-model request rules

- At least two requested model names must resolve to distinct registry IDs.
- At least two models must complete successfully for an agreement result.
- `mean`, `max`, or supported `native` may be used for sequence-level methods.
- The cross-model representation must also appear in `representations`.
- Residue CKA/RSA requires `residue` in `representations`.
- The output always retains the guardrail that raw cross-PLM vectors were not
  directly compared.

## FASTA input

FASTA parsing is available directly:

```python
from pathlib import Path

from plms_framework import EmbeddingRequest, Pipeline
from plms_framework.sequences import parse_fasta

sequences = parse_fasta(Path("examples/proteins.fasta").read_text())
request = EmbeddingRequest(models=("reference",), sequences=sequences)
result = Pipeline().embed(request)
```

The complete text after `>` becomes the sequence ID. Blank lines are ignored.
A sequence before the first header, an empty header, or a FASTA file without
records is rejected. Normal sequence canonicalization still occurs when the
pipeline runs.

## Understanding results

`embed()` and `compare()` return JSON-serializable dictionaries. The main keys
are:

| Key | Meaning |
|---|---|
| `schema_version` | Result contract version |
| `sequence_order` | Stable row/column order for all matrices |
| `models` | Successful canonical model IDs and their outputs |
| `failures` | Structured per-model failures when partial execution is enabled |
| `provenance` | Framework, platform, timing, execution, and sequence digests |
| `sequence_baselines` | Sequence-only comparison matrices; compare results only |
| `cross_model` | Cross-PLM results or `{"enabled": false}` |
| `requested_visualizations` | Rendering instructions retained by `compare()` |

Each successful model contains:

| Key | Meaning |
|---|---|
| `manifest` | Model identity, revision, capabilities, license, and limitations |
| `availability` | Adapter availability result at execution time |
| `representations` | One record per sequence, method, and requested layer |
| `matrices` | Within-model pooled-vector metric matrices |
| `residue_pairs` | Requested residue-level results by `left_id::right_id` |
| `provenance` | Artifact, adapter, revision, device, and backend data |

A representation record contains `sequence_id`, `method`, `layer`, `values`,
`shape`, `norm_before`, `normalization`, and `metadata`.

Always use `sequence_order` for matrix row and column labels. Do not rely on
sorting model or sequence IDs yourself.

### Save a numerical result

```python
import json
from pathlib import Path

Path("comparison.json").write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
)
```

Residue representations can make JSON files very large. Avoid requesting them
when only protein-level vectors are needed.

## Rendering visualizations

`Pipeline.compare()` records visualization choices and produces the numerical
data needed to render them. In direct Python use, rendering is an explicit step:

```python
from plms_framework.artifacts import ArtifactStore
from plms_framework.visualization import render_requested

store = ArtifactStore("analysis_artifacts")
artifacts = render_requested(result, store, job_id="run-001")

for artifact in artifacts:
    print(artifact["name"], artifact["sha256"])
```

Request visualizations in the comparison dictionary:

```python
"visualizations": ["clustered_heatmap", "distance_distribution"]
```

Supported names and requirements:

| Name | File | Required data/dependency |
|---|---|---|
| `clustered_heatmap` | SVG | A pairwise matrix |
| `distance_distribution` | SVG | A pairwise matrix |
| `nearest_neighbor_graph` | SVG | A pairwise matrix; shows top three neighbors |
| `residue_similarity_map` | SVG per pair | Residue representation and residue metric |
| `model_agreement` | SVG | Available cross-model matrix Spearman result |
| `model_matrix_small_multiples` | SVG per model | Cross-model relationship matrices |
| `consensus_heatmap` | SVG | `consensus_rank` method |
| `model_disagreement` | SVG | `pair_disagreement` method |
| `residue_model_agreement` | SVG per sequence | `residue_cka` or `residue_rsa` |
| `pcoa` | HTML | Pairwise matrix plus `visual` extra |
| `umap` | HTML | Pairwise matrix plus `visual` extra |
| `dendrogram` | HTML | Pairwise matrix plus `visual` extra |

Dependency-light plots are SVG. PCoA, UMAP, and dendrogram use optional
scientific/Plotly dependencies. UMAP is exploratory and can visually exaggerate
gaps; use quantitative matrices for conclusions.

The renderer returns artifact descriptors containing the filename, byte size,
SHA-256 checksum, and media type. Rendering problems are recorded in
`visualization_warnings.json` rather than discarding successful numerical work.

## Importing existing embeddings

Precomputed embeddings can participate in the same diagnostics,
transformations, annotation analysis, and cross-model functions without running
a checkpoint:

```python
from plms_framework.external import build_imported_space

imported = build_imported_space(
    space_name="external-esm-run",
    source={
        "display_name": "ESM embeddings from cluster",
        "family": "ESM-2",
        "provider": "example-lab",
        "artifact": "facebook/esm2_t33_650M_UR50D",
        "revision": "record-an-immutable-revision",
        "license": "record-upstream-license",
        "citation": "record-model-citation",
    },
    representations=[
        {
            "method": "mean",
            "layer": 33,
            "normalization": "l2",
            "embeddings": [
                {"sequence_id": "a", "values": [0.1, 0.2, -0.3]},
                {"sequence_id": "b", "values": [0.4, -0.1, 0.2]},
                {"sequence_id": "c", "values": [-0.2, 0.3, 0.1]},
            ],
        }
    ],
)
```

Supported imported methods are `mean`, `max`, `native`, and `residue`.
Supported normalization declarations are:

- `unknown`
- `none`
- `l2`
- `mean_centered`
- `standardized`
- `whitened`
- `custom`

The declaration records what the caller says was already done; import does not
normalize the supplied numbers.

Import validation requires non-empty unique IDs, finite numbers, consistent
feature dimensions, rectangular residue matrices, and identical ID ordering
across all representations. Default safety limits are 10,000 records, 32,768
features, and 5,000,000 total scalar values. They can be made more restrictive
or deliberately increased with `max_records`, `max_dimension`, and
`max_scalar_values`.

Imported metadata is provenance, not verification. The framework validates the
structure of the values but does not prove that they came from the stated
checkpoint.

### Compare two imported spaces

`compare_cross_model()` consumes the same model-output contract returned by
imports and the pipeline:

```python
from plms_framework import CrossModelSpec
from plms_framework.cross_model import compare_cross_model
from plms_framework.models import CrossModelMethod, RepresentationMethod

left = build_imported_space(
    space_name="space-left",
    representations=[
        {
            "method": "mean",
            "embeddings": [
                {"sequence_id": "a", "values": [1.0, 0.0]},
                {"sequence_id": "b", "values": [0.8, 0.2]},
                {"sequence_id": "c", "values": [0.0, 1.0]},
            ],
        }
    ],
)
right = build_imported_space(
    space_name="space-right",
    representations=[
        {
            "method": "mean",
            "embeddings": [
                {"sequence_id": "a", "values": [2.0, 0.0, 0.0]},
                {"sequence_id": "b", "values": [1.5, 0.5, 0.0]},
                {"sequence_id": "c", "values": [0.0, 2.0, 0.0]},
            ],
        }
    ],
)

outputs = {**left["models"], **right["models"]}
sequence_order = left["sequence_order"]

comparison = compare_cross_model(
    outputs,
    sequence_order,
    CrossModelSpec(
        enabled=True,
        methods=(
            CrossModelMethod.MATRIX_SPEARMAN,
            CrossModelMethod.LINEAR_CKA,
        ),
        representation=RepresentationMethod.MEAN,
    ),
)
```

The spaces may have different feature dimensions because raw coordinates are
not compared directly. They must contain the same sequence IDs in the same
order.

## Embedding diagnostics

Diagnostics consume successful model outputs from `embed()`, `compare()`, an
import, or a transformation:

```python
from plms_framework import DiagnosticSpec
from plms_framework.diagnostics import diagnose_embeddings
from plms_framework.models import RepresentationMethod

diagnostics = diagnose_embeddings(
    result["models"],
    result["sequence_order"],
    DiagnosticSpec(
        representation=RepresentationMethod.MEAN,
        layer=None,
        sample_pairs=10_000,
        hubness_k=10,
        seed=42,
        max_exact_hubness_sequences=2_000,
    ),
)
```

Supported representations are `mean`, `max`, and `native`. Options:

| Option | Default | Meaning |
|---|---:|---|
| `representation` | `mean` | Pooled vectors to diagnose |
| `layer` | `None` | Exact layer, or first complete matching layer |
| `sample_pairs` | 10,000 | Maximum reproducibly sampled distinct pairs |
| `hubness_k` | 10 | Neighbors used for k-occurrence hubness |
| `seed` | 42 | Pair-sampling seed |
| `max_exact_hubness_sequences` | 2,000 | Above this, hubness is reported unavailable |

Results cover norms, zero and duplicate vectors, per-dimension means and
variances, participation ratio, raw and mean-centered average cosine,
anisotropy, cosine-distance concentration, and exact hubness. These describe
geometry; they do not establish biological quality.

## Mean-centering

Mean-centering creates new model-like outputs and leaves the source dictionary
unchanged:

```python
from plms_framework import TransformationSpec
from plms_framework.models import RepresentationMethod, TransformationMethod
from plms_framework.transformations import transform_embeddings

centered = transform_embeddings(
    result["models"],
    result["sequence_order"],
    TransformationSpec(
        method=TransformationMethod.MEAN_CENTER,
        representation=RepresentationMethod.MEAN,
        layer=None,
        fit_sequence_ids=("a", "b"),
        sample_pairs=10_000,
        hubness_k=10,
        seed=42,
    ),
)
```

Supported representations are `mean`, `max`, and `native`; the only current
transformation is `mean_center`. `fit_sequence_ids=None` fits on all sequences.
For held-out evaluation, explicitly provide only training/reference IDs to
avoid leakage.

The result records the fitted mean, fitting IDs, parameter checksum, source to
derived model mapping, and geometry diagnostics before and after centering. A
derived model ID ends with `::mean-centered`.

## Categorical annotation analysis

This analysis asks whether supplied categorical labels are locally organized in
a model's cosine-neighbor space:

```python
from plms_framework import AnnotationAnalysisSpec
from plms_framework.annotations import analyze_categorical_annotations
from plms_framework.models import MissingAnnotationPolicy, RepresentationMethod

annotations = [
    {"sequence_id": "a", "categorical": {"family": "kinase"}},
    {"sequence_id": "b", "categorical": {"family": "kinase"}},
    {"sequence_id": "c", "categorical": {"family": "other"}},
]

analysis = analyze_categorical_annotations(
    result["models"],
    result["sequence_order"],
    annotations,
    AnnotationAnalysisSpec(
        feature="family",
        representation=RepresentationMethod.MEAN,
        layer=None,
        k=(1, 2),
        missing_policy=MissingAnnotationPolicy.ERROR,
        max_exact_sequences=2_000,
    ),
)
```

Supported representations are `mean`, `max`, and `native`. Categorical values
may be strings, integers, or booleans. At least two labeled sequences and two
classes are required.

Options:

| Option | Default | Meaning |
|---|---|---|
| `feature` | Empty; must be supplied | Key selected from each `categorical` mapping |
| `representation` | `mean` | Pooled vector method |
| `layer` | `None` | Exact layer, or first complete matching layer |
| `k` | `(1, 5, 10)` | Positive neighbor counts; values above `N - 1` are reduced |
| `missing_policy` | `error` | `error` rejects missing labels; `drop` excludes them |
| `max_exact_sequences` | 2,000 | Exact-analysis safety threshold |

Outputs include per-sequence neighbor IDs, micro/macro consistency, chance
baseline, chance-adjusted consistency, precision and recall at k, per-class
summaries, and class-mixing matrices. These are descriptive in-sample results,
not evidence of held-out generalization.

## Errors and partial failures

Framework exceptions have stable codes and structured context:

```python
from plms_framework.errors import FrameworkError

try:
    result = Pipeline().compare(request)
except FrameworkError as exc:
    print(exc.as_dict())
```

| Exception | Code | Typical cause |
|---|---|---|
| `ValidationError` | `validation_error` | Invalid sequence, option, region, or request combination |
| `CapabilityError` | `unsupported_capability` | Model does not support the requested representation/layer |
| `DependencyUnavailableError` | `dependency_missing` | Required backend package is absent |
| `ModelUnavailableError` | `model_unavailable` | Checkpoint download/load or backend compatibility failed |
| `ResourceLimitError` | `resource_limit` | Length or exact-comparison limit exceeded |

With `allow_partial=True`, failures from one requested model are placed in
`result["failures"]` while other models continue:

```python
for failure in result["failures"]:
    print(failure["model"], failure["code"], failure["message"])
```

With `allow_partial=False`, the first model failure is raised immediately. Use
the latter for strict batch pipelines where any missing model invalidates the
whole run.

## Custom registries

`Pipeline` accepts a custom `ModelRegistry`. This is useful for private adapters,
tests, or a restricted production deployment:

```python
from dataclasses import replace

from plms_framework import Pipeline, default_registry
from plms_framework.adapters import DeterministicAdapter
from plms_framework.registry import ModelRegistry

registry = ModelRegistry()
reference_manifest = default_registry.manifest("reference")

registry.register(
    replace(reference_manifest, id="reference-small", aliases=()),
    DeterministicAdapter,
)

pipeline = Pipeline(registry)
```

Register adapter factories/classes, not already-loaded model objects. This keeps
discovery cheap and lets loading remain lazy. Custom adapters must return exactly
one residue row for every input amino acid and must not silently pool, normalize,
truncate, or replace residues.

## Reproducibility and resource guidance

- Keep the returned model manifest and run provenance with every analysis.
- Prefer immutable checkpoint revisions. Most registry entries still use an
  upstream branch name until they receive checkpoint-specific conformance work.
- Record the selected layer, pooling, normalization, device, and precision.
- Inspect `result["failures"]`; a partial result is not the same as a complete
  multi-model experiment.
- Use `Pipeline.estimate()` before large runs, but also account for checkpoint
  parameters, activations, hidden states, and tokenizer memory.
- Pairwise matrices scale as `N squared`; residue-pair comparisons can scale as
  `L1 x L2` for every sequence pair.
- Avoid requesting residue outputs, intermediate layers, or duplicate metrics
  unless the analysis needs them.
- Do not treat attractive clusters, low anisotropy, or high label consistency as
  proof of biological validity. Use appropriate held-out biological benchmarks.
- Never compare raw vectors from different PLMs directly. Use the implemented
  relationship-matrix, CKA, neighbor, rank, or residue-observation methods.

For HTTP usage, asynchronous jobs, and endpoint payloads, see the project
[`README.md`](../README.md). For implementation priorities and known missing
features, see [`ROADMAP.md`](../ROADMAP.md).
