# Using PLMs Framework from Python

PLMs Framework can be used as a normal Python library. You do not need to run
the FastAPI server when the embedding and comparison are part of a notebook,
analysis script, training pipeline, or another Python application.

For a small runnable comparison of two raw sequences, start with
[`examples/test_two_seqs.py`](../examples/test_two_seqs.py). It uses the same
metrics as the [noisy FASTA analyser](NOISY_SEQUENCE_ANALYSER.md). The
[workflow guide](WORKFLOWS.md) lists script flags and CLI commands.

This guide documents the currently implemented Python interface, every request
option accepted by the core pipeline, the result structure, and direct Python
access to diagnostics, transformations, annotation analysis, imported
embeddings, FASTA parsing, and visualization rendering.

## Contents

- [Installation](#installation)
- [Choose between Python and the API](#choose-between-python-and-the-api)
- [Quick start](#quick-start)
- [Dictionary requests and typed requests](#dictionary-requests-and-typed-requests)
- [Complete request option reference](#complete-request-option-reference)
- [Discovering models](#discovering-models)
- [Model and backend guide](#model-and-backend-guide)
- [ESM-C setup and troubleshooting](#esm-c-setup-and-troubleshooting)
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

The core library requires Python 3.11 or newer. The complete installation,
including ESM-C, requires Python 3.12 or newer. From the repository root, the
recommended one-file installation is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the package and its `all` runtime profile: the API,
Hugging Face models, ESM-C, and advanced visualizations. It refers back to
`pyproject.toml`, which remains the single source of dependency versions. Model
weights are not installed eagerly; each checkpoint is downloaded when that
model is first loaded.

For a smaller editable installation on Python 3.11 or newer, install only the
core package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core installation has no mandatory third-party dependencies and includes
the deterministic reference adapter. Install the Hugging Face dependencies to
run ESM-1, ESM-2, ProtTrans, Ankh, and ProGen2 checkpoints:

```bash
python -m pip install -e '.[hf]'
```

ESM-C uses Biohub's maintained `esm` implementation instead of the generic
adapter. Its current dependency stack requires Python 3.12 or newer. Create the
environment with a 3.12+ interpreter and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[esmc]'
```

Install advanced interactive visualizations with:

```bash
python -m pip install -e '.[visual]'
```

The API dependencies are not needed for direct Python use. On Python 3.12 or
newer, install all runtime features only when they are required:

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

For ESM-C, install the separate `esmc` extra and make failures raise while
checking a new installation:

```python
request = request_from_dict(
    {
        "models": ["esmc-300m"],
        "sequences": [{"id": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"}],
        "representations": [{"method": "mean", "normalize": "l2"}],
        "execution": {
            "device": "cpu",
            "precision": "float32",
            "allow_partial": False,
        },
    },
    compare=False,
)

result = Pipeline().embed(request)
```

Use `bfloat16` on a compatible CUDA device. ESM-C does not advertise MPS or
float16 in this framework.

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

## Complete request option reference

`EmbeddingRequest` is the input to `embed()` and `estimate()`.
`CompareRequest` adds comparison-only fields and is the input to `compare()`.
The following table is the complete top-level contract:

| Field | Embed/estimate | Compare | Default | Meaning |
|---|---:|---:|---|---|
| `models` | Yes | Yes | Required | One or more framework IDs or aliases |
| `sequences` | Yes | Yes | Required | Protein records; at least one |
| `representations` | Yes | Yes | One L2-normalized `mean` | Outputs to materialize for every successful model |
| `execution` | Yes | Yes | `ExecutionOptions()` | Device, precision, failure, length, and scaling policy |
| `metrics` | No | Yes | `cosine_similarity` | Pairwise vector, sequence, or residue calculations |
| `visualizations` | No | Yes | Empty | Rendering instructions retained in the numerical result |
| `cross_model` | No | Yes | Disabled | Relationship-based comparison between successful PLMs |

Important interactions:

- `models` selects adapters; aliases are resolved before execution and successful
  output keys always use canonical IDs.
- A representation is computed for every sequence and every requested model.
  Requesting two layers causes two forward passes per model in the current
  implementation.
- Each method/layer combination must be unique. Normalization is not part of
  result matrix keys, so request a method/layer only once.
- Vector metrics require at least one of `mean`, `max`, or a supported `native`
  representation. Residue metrics require `residue`.
- `visualizations` do not render files in `Pipeline.compare()`; call
  `render_requested()` afterward.
- Sequence-only metrics do not require a model representation, but model
  embedding still runs for the requested models.
- Global option errors are raised before model execution. Model-specific
  capability/load errors follow `allow_partial`.

`cross_model` has these fields:

| Field | Values | Default | Meaning |
|---|---|---|---|
| `enabled` | Boolean | `False` | Enables cross-PLM analysis |
| `methods` | Cross-model method names | `matrix_spearman` | Analyses to run |
| `representation` | `mean`, `max`, or supported `native` | `mean` | Pooled space used by sequence-level methods |
| `neighbor_k` | Positive integers | `(1, 5, 10)` | Neighborhood sizes; reduced to `N - 1` when necessary |
| `max_disagreements` | Positive integer | `20` | Maximum ranked disagreement pairs returned |

All accepted representation, execution, metric, visualization, and cross-model
method values are detailed in the corresponding sections below. There is no
special `"all"` model selector: enumerate model IDs deliberately so a request
cannot trigger unbounded downloads.

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

| Model ID | Backend | Layers | Dimension | Native residues | Conformance status |
|---|---|---:|---:|---:|---|
| `reference-deterministic-v1` | Built in | 0 | 16 | Unbounded | Tested; not a biological PLM |
| `esm2-8m` | Hugging Face | 6 | 320 | 1,022 | Tested for CPU shapes, masking, and batch invariance |
| `esm2-35m` | Hugging Face | 12 | 480 | 1,022 | Checkpoint validation required |
| `esm2-150m` | Hugging Face | 30 | 640 | 1,022 | Checkpoint validation required |
| `esm2-650m` | Hugging Face | 33 | 1,280 | 1,022 | Checkpoint validation required |
| `esm2-3b` | Hugging Face | 36 | 2,560 | 1,022 | Checkpoint validation required |
| `esm2-15b` | Hugging Face | 48 | 5,120 | 1,022 | Checkpoint validation required |
| `esm1b-650m` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esm1v-650m-1` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esm1v-650m-2` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esm1v-650m-3` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esm1v-650m-4` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esm1v-650m-5` | Hugging Face | 33 | 1,280 | 1,022 | Pinned; checkpoint validation required |
| `esmc-300m` | Biohub `esm` | 30 | 960 | 2,046 | Adapter implemented; checkpoint validation required |
| `esmc-600m` | Biohub `esm` | 36 | 1,152 | 2,046 | Adapter implemented; checkpoint validation required |
| `esmc-6b` | Biohub `esm` | 80 | 2,560 | 2,046 | Adapter implemented; checkpoint validation required |
| `prott5-xl-uniref50` | Hugging Face | 24 | 1,024 | Not declared | Checkpoint validation required |
| `prott5-xl-bfd` | Hugging Face | 24 | 1,024 | Not declared | Pinned; checkpoint validation required |
| `protbert` | Hugging Face | 30 | 1,024 | Not declared | Pinned; checkpoint validation required |
| `protbert-bfd` | Hugging Face | 30 | 1,024 | Not declared | Checkpoint validation required |
| `protalbert` | Hugging Face | 12 | 4,096 | Not declared | Pinned; checkpoint validation required |
| `protxlnet` | Hugging Face | 30 | 1,024 | Not declared | Pinned; checkpoint validation required |
| `ankh-base` | Hugging Face | Not declared | 768 | Not declared | Checkpoint validation required |
| `ankh-large` | Hugging Face | Not declared | 1,536 | Not declared | Checkpoint validation required |
| `progen2-small` | Hugging Face custom code | 12 | 1,024 | 1,022 | Pinned community conversion; checkpoint validation required |
| `progen2-medium` | Hugging Face custom code | 27 | 1,536 | 1,022 | Pinned community conversion; checkpoint validation required |

Model aliases include `reference`, upstream ESM-1/ESM-2 checkpoint names,
Biohub ESM-C forms such as `esmc_300m` and `biohub/ESMC-300M`, ProtTrans
underscore forms such as `prott5_xl_bfd`, and `progen2_small`/
`progen2_medium`.

`adapter.availability().available` currently means that the adapter's required
Python packages are importable. It does not prove that a checkpoint is cached,
downloadable, licensed for the intended use, compatible with the adapter, or
small enough for the selected hardware. The manifest's `conformance` field and
`limitations` list provide the stronger qualification.

The ESM-C entries use their own Biohub adapter and never fall through to the
generic Hugging Face loader. They have accurate layer/dimension metadata and a
2,046-residue limit, but they have not yet passed the repository's real-model
conformance suite. Treat the status literally.

## Model and backend guide

| Family | Install | Input handling | Practical notes |
|---|---|---|---|
| Reference | Core package | One character per residue | Deterministic software fixture only; never use as a biological model |
| ESM-1b/1v | `.[hf]` | Raw sequence through the checkpoint tokenizer | Six pinned 650M checkpoints; registered but not conformance-tested here |
| ESM-2 | `.[hf]` | Raw sequence through the checkpoint tokenizer | `esm2-8m` is the only checkpoint with a checked-in real-model conformance test |
| ESM-C | `.[esmc]` | Biohub character tokenizer with `<cls>`/`<eos>` removed | Python 3.12+; CPU/CUDA; 2,046 residues; current adapter uses portable SDPA |
| ProtT5 | `.[hf]` | Residues are space-separated before tokenization | UniRef50 and BFD XL checkpoints; encoder output only; large checkpoints |
| ProtBert | `.[hf]` | Residues are space-separated before tokenization | UniRef100 and BFD checkpoints; not conformance-tested here |
| ProtAlbert | `.[hf]` | Residues are space-separated before tokenization | 4,096-dimensional output; sparse upstream model metadata |
| ProtXLNet | `.[hf]` | Residues are space-separated before tokenization | XLNet base-model output; sparse upstream model metadata |
| Ankh | `.[hf]` | Raw sequence; encoder side of encoder-decoder model | Non-commercial upstream license; verify terms and checkpoint behavior |
| ProGen2 | `.[hf]` | Raw sequence through a causal tokenizer | Pinned community conversions; executes remote model code; residue states encode left context only |

The Hugging Face adapter selects a base encoder for ESM, BERT, ALBERT, and
XLNet; the encoder side of sequence-to-sequence checkpoints for ProtT5 and
Ankh; and a causal-LM loader for ProGen2. Special and padding tokens are removed
with tokenizer-provided masks. A checkpoint fails rather than returning an
embedding when the remaining token count is not exactly the sequence length.

ProGen2 is an explicit security exception. Its manifests set
`trust_remote_code=true` because the registered `hugohrban` conversions provide
custom configuration and modeling code. Both revisions are immutable pins, but
deployments should still review that code and upstream licensing before use.

Run all newly registered checkpoints on a capable machine with:

```bash
HF_HOME=.model_cache PLMS_RUN_EXTENDED_MODEL_TESTS=1 \
PLMS_TEST_DEVICE=cuda PLMS_TEST_PRECISION=float16 \
  python -m unittest tests.test_real_models.ExtendedRealModelTests -v
```

This opt-in suite checks loading and mean/residue output shapes for all six
ESM-1 checkpoints, the four added ProtTrans checkpoints, and both ProGen2
checkpoints. It is skipped by the normal dependency-free test suite. Set
`PLMS_TEST_MODELS` to comma-separated IDs or aliases to run a smaller subset.

`mean` is intentionally the same framework-controlled operation for every
family: average exactly one returned vector per native residue, then apply the
requested normalization. Model-specific classification tokens and upstream
poolers are not substituted for it.

The registry does not promise that every entry will run on one machine. It
describes the adapter and checkpoint; dependency versions, network access,
disk, RAM/VRAM, dtype support, and upstream terms still determine whether an
individual run can load.

## ESM-C setup and troubleshooting

ESM-C has a dedicated dependency and execution path. A reliable diagnostic is:

```bash
python --version
python -m pip show esm torch plms-framework
python -m plms_framework.cli models
```

Then run one short sequence with `allow_partial=False`, `device="cpu"`, and
`precision="float32"`. This makes the original exception visible instead of
returning it inside `result["failures"]`.

Common ESM-C failures:

| Symptom | Cause | Action |
|---|---|---|
| `python_version_unsupported` | Current Biohub `esm` requires Python 3.12+ | Create a Python 3.12+ environment |
| `dependency_missing` | `torch` or Biohub `esm` is absent | Install `pip install -e '.[esmc]'` |
| `dependency_incompatible` | Old Biohub `esm`, or another distribution owns the `esm` module | Remove the conflict and install the declared extra; `fair-esm` is not the ESM-C backend |
| `model_unavailable` during download | No network/cache access, insufficient disk, or upstream Hub error | Check the nested message and pre-populate the cache if running offline |
| CUDA requested but unavailable | CPU-only PyTorch or inaccessible GPU | Use `device="cpu"` or install a matching CUDA PyTorch build |
| Out of memory | Checkpoint, activations, batch, or hidden-state request is too large | Start with `esmc-300m`, one short sequence, pooled output, and no intermediate layer |
| MPS or float16 capability failure | ESM-C does not advertise that execution combination here | Use CPU/float32 or compatible CUDA/bfloat16 |
| Length error above 2,046 residues | The 2,048-token context also contains `<cls>` and `<eos>` | Shorten upstream or choose another model; framework truncation/windowing is not implemented |

The adapter uses `biohub/ESMC-*` weights through `EsmcForMaskedLM`, removes
boundary and padding tokens with tokenizer masks, and records Biohub `esm`
version, resolved dtype, device, and attention implementation in provenance.
Published checkpoint storage is float32, so downloads and CPU memory can be
substantially larger than half-precision storage; the ESM-C resource estimator
therefore uses four bytes per declared parameter. Inference can also need
significant activation memory.

An opt-in real-checkpoint shape test is included. It downloads and loads the
300M checkpoint, so run it only in a suitable Python 3.12+ environment:

```bash
PYTHONPATH=src PLMS_RUN_ESMC_TESTS=1 \
  python -m unittest tests.test_real_models.RealESMCTests -v
```

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

Method-specific rules:

- `mean` and `max` return one vector per sequence.
- `residue` returns one row per canonical amino acid. With `normalize="l2"`,
  normalization is applied independently to every row.
- `region_mean` ignores `window_size`/`window_step` and requires at least one
  valid region on every sequence.
- `window_mean` ignores named regions, requires `window_size`, and starts
  windows at residues `1, 1 + window_step, ...`. The final partial window is
  retained; processing stops after a window reaches the sequence end.
- `native` requires explicit model support. None of the default live-model
  manifests currently advertises it.
- Options irrelevant to the selected method are accepted by the dataclass but
  have no numerical effect; omit them to keep requests unambiguous.

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
layers. It selects the adapter's final representation without requesting the
complete hidden-state stack. The only supported string is exactly
`"model_default"`; names such as `"last"` are rejected.

For adapters with a known layer count `K`, integer layers range from `0` through
`K`. Layer `0` is the input embedding state and layer `K` is the final
transformer output under the Hugging Face-style hidden-state convention. An
integer asks the backend for hidden states and can substantially increase peak
memory. The reference fixture has `K = 0`.

Do not compare matrices built from different layers as if they were the same
representation. The matrix key records both method and layer.

## Execution options

`execution` controls inference and safety behavior:

| Option | Supported/current values | Default | Behavior |
|---|---|---|---|
| `device` | `auto`, `cpu`, `cuda`, `mps` | `auto` | Explicit values must also be advertised by every selected model |
| `precision` | `auto`, `float32`, `float16`/`fp16`, `bfloat16`/`bf16` | `auto` | Aliases are equivalent; explicit values must be advertised by the model |
| `all_to_all` | `exact` | `exact` | Approximate comparison is not implemented |
| `on_long_sequence` | `error` | `error` | Windowed model inference is not implemented |
| `allow_partial` | Boolean | `True` | Isolate per-model failures instead of aborting a multi-model run |
| `max_exact_sequences` | Integer at least 2 | `2000` | Maximum sequence count accepted by `compare()`; does not limit `embed()` |

The dataclass layer stores `device`, `precision`, `all_to_all`, and
`on_long_sequence` as strings, but the pipeline validates the implemented
values before inference. A misspelling or future value raises `ValidationError`;
it never silently activates, disables, or substitutes an algorithm.

Backend defaults and constraints:

| Backend | `device="auto"` | `precision="auto"` | Important constraint |
|---|---|---|---|
| Reference | CPU | Internal deterministic arithmetic | Explicit neural-model precisions are not advertised |
| Hugging Face | CUDA, then MPS, then CPU | Checkpoint/default dtype | Runtime support still depends on the checkpoint and PyTorch device |
| Biohub ESM-C | CUDA, otherwise CPU | bfloat16 on CUDA; float32 on CPU | Only CPU/CUDA and float32/bfloat16 are advertised |

`allow_partial` applies after a model has been selected. Unknown global option
values and invalid request-wide combinations always raise. With partial mode,
a model-specific unsupported device, precision, layer, length, dependency, or
load problem is recorded and the next model is attempted. If every model fails,
the method can return `models={}` with a non-empty `failures` list; callers must
check it.

Example:

```python
"execution": {
    "device": "auto",
    "precision": "auto",
    "all_to_all": "exact",
    "on_long_sequence": "error",
    "allow_partial": True,
    "max_exact_sequences": 1000,
}
```

Set explicit hardware/dtype values only after inspecting the selected model's
`devices` and `precisions` manifest fields. For example, CUDA/float16 is an
advertised Hugging Face combination, whereas ESM-C advertises CUDA/bfloat16.

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
| `cosine_similarity` | Higher is closer | Dot product divided by norms; clamped to `[-1, 1]`; a zero vector has similarity `0` |
| `cosine_distance` | Lower is closer | `1 - cosine_similarity`; range `[0, 2]` |
| `angular_distance` | Lower is closer | `acos(cosine) / pi`; range `[0, 1]` |
| `correlation_distance` | Lower is closer | Cosine distance after subtracting each vector's feature mean; a constant centered vector follows the zero-vector convention |
| `euclidean_distance` | Lower is closer | Scale-sensitive L2 distance; no bounded range |

For L2-normalized vectors, cosine and Euclidean distance encode the same pairwise
ordering. Request both only when their different numerical scales help a
downstream consumer.

### Supported sequence metrics

Sequence metrics use the amino-acid strings and are reported once under
`sequence_baselines`, not repeated for each model:

| Metric | Direction | Current definition |
|---|---|---|
| `length_ratio` | Higher is closer | Shorter length divided by longer length; range `(0, 1]` for accepted non-empty sequences |
| `sequence_identity` | Higher is closer | Global Needleman-Wunsch; match `+1`, mismatch `-1`, linear gap `-1`; matches divided by alignment columns |
| `kmer_jaccard` | Higher is closer | Jaccard similarity of sets of unique 3-mers; `k` is fixed at 3 in requests |
| `blosum62_global` | Higher is closer | Global BLOSUM62, linear gap `-4`, divided by geometric mean of self scores |
| `blosum62_local` | Higher is closer | Local Smith-Waterman BLOSUM62 with linear gap `-4`, normalized by self scores |

The BLOSUM table has explicit `B`, `Z`, and `X` entries. `J`, `U`, and `O` are
mapped symmetrically to `X` for these two metrics; model tokenization remains
unchanged.

### Supported residue metrics

These require `{"method": "residue"}` in the same request:

| Metric | Result |
|---|---|
| `residue_global_alignment` | Global alignment over residue cosine scores; fixed linear gap `-0.5` |
| `residue_local_alignment` | Smith-Waterman-style local alignment over residue cosine scores; fixed linear gap `-0.5` |
| `residue_best_match` | Average of left-to-right and right-to-left mean best-residue cosine similarities |
| `residue_optimal_transport` | Balanced entropy-regularized Sinkhorn similarity; uniform residue mass, epsilon `0.1`, 100 iterations |

Residue work can be much larger than pooled-vector comparison. Comparing
proteins of lengths `L1` and `L2` constructs an `L1 x L2` residue-similarity
matrix for that pair.

Alignment results include the raw score, score divided by path length, mean
matched cosine, gap penalty, and a 1-based path using `None` for gaps. Optimal
transport returns similarity, cost, transported mass, epsilon, iteration count,
and method name. These algorithm parameters are currently fixed rather than
request options. Request exactly one `residue` method/layer when using residue
metrics, because `residue_pairs` has no layer axis.

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
- Sequence-level methods require exactly one requested layer of the selected
  pooled method; residue methods require exactly one requested residue layer.
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

Complete `build_imported_space()` arguments:

| Argument | Default | Meaning |
|---|---|---|
| `space_name` | Required | 1-128 safe characters: letters, digits, `.`, `_`, `:`, `-`; first character alphanumeric |
| `representations` | Required | Non-empty imported representation list |
| `source` | `None` | Caller-declared provenance fields; recorded but not verified |
| `max_records` | `10_000` | Per-representation record limit |
| `max_dimension` | `32_768` | Maximum feature width |
| `max_scalar_values` | `5_000_000` | Limit across all supplied vectors/matrices |

Each representation accepts `method` (default `mean`), `layer` (default
`external`), `normalization` (default `unknown`), and a required `embeddings`
list. Each embedding requires `sequence_id` and `values`, with optional
`metadata`. A method/layer pair must be unique.

Recognized `source` fields are `display_name`, `family`, `provider`, `artifact`,
`revision`, `license`, `citation`, `architecture`, `tokenizer_convention`,
`one_token_per_residue`, `special_tokens`, `parameters`, `preprocess`, and
`trust_remote_code`.

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

| Option | Default | Meaning |
|---|---|---|
| `method` | `mean_center` | Transformation; currently the only value |
| `representation` | `mean` | `mean`, `max`, or supported `native` source vectors |
| `layer` | `None` | Exact layer, or first complete matching layer with a warning |
| `fit_sequence_ids` | `None` | Unique fitting subset; `None` means all source IDs |
| `sample_pairs` | `10_000` | Pair sample used for before/after diagnostics |
| `hubness_k` | `10` | Neighborhood size used for diagnostics |
| `seed` | `42` | Reproducible diagnostic pair-sampling seed |

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

Unexpected third-party exceptions are converted to a per-model failure with
code `unexpected_model_failure` only when partial mode is enabled. If strict
mode is enabled, the original exception is raised. Backend adapters wrap known
checkpoint/dependency problems in the structured exceptions above.

With `allow_partial=True`, failures from one requested model are placed in
`result["failures"]` while other models continue:

```python
for failure in result["failures"]:
    print(failure["model"], failure["code"], failure["message"])
```

With `allow_partial=False`, the first model failure is raised immediately. Use
the latter for strict batch pipelines where any missing model invalidates the
whole run and while diagnosing a new backend. Invalid request-wide execution
values are raised before partial handling. A partial run can legitimately have
an empty `models` mapping when every requested model failed.

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
