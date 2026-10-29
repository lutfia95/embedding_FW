# PLMs Framework

[![Tests](https://github.com/lutfia95/embedding_FW/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lutfia95/embedding_FW/actions/workflows/ci.yml)
[![Docs](https://github.com/lutfia95/embedding_FW/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/lutfia95/embedding_FW/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-D22128)](LICENSE)

PLMs Framework is a Python library and HTTP API for generating, comparing,
and visualizing protein language model (PLM) embeddings. It also provides
embedding diagnostics and an inference timing benchmark. A shared model registry
and adapter interface handle differences in tokenization, sequence limits,
embedding dimensions, and backend requirements.

For a quick two-protein comparison, see the
[runnable example](examples/test_two_seqs.py). For reference-to-noisy FASTA
scoring, see the [analyser guide](doc/NOISY_SEQUENCE_ANALYSER.md). The
[workflow guide](doc/WORKFLOWS.md) lists commands and their options.

## Comparison principles

The framework uses these conventions:

1. **Mean pooling is the portable default.** It averages valid residue vectors,
   excluding padding and boundary tokens.
2. **Residue representations remain available.** They support motif discovery,
   local similarity, embedding-based alignment, and residue-level tasks.
3. **Raw vectors are compared only within the same model and representation.**
   ESM-2 and ProtT5 coordinates cannot be compared directly.
4. **Models are compared through relationships.** Pairwise matrices, rankings,
   neighbor overlap, and representational-similarity methods compare models
   without assuming that their embedding coordinates match.
5. **Execution conditions are reported.** Truncation policy, unavailable
   checkpoints, approximation, partial failures, normalization, and adapter
   limitations appear in structured output.

## Architecture

```text
API / CLI / Python
       |
       v
request validation -> capability resolution -> resource estimate
       |                       |
       |                       v
       |                 model registry
       |                       |
       v                       v
sequence canonicalization -> model adapter -> residue representations
                                      |
                                      v
                        pooling and normalization
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
          within-model matrices                  residue comparisons
                    |                         alignment / best match / OT
                    +-----------------+------------------+
                                      |
                                      v
                  cross-model agreement and visualizations
                                      |
                                      v
                  result + artifacts + provenance manifest
```

The core package has no mandatory third-party dependency. Real neural models,
the web API, and advanced visualizations are optional installation profiles and
are imported lazily.

For direct use from notebooks and Python applications, including complete
option tables and typed examples, see
[`doc/PYTHON_USAGE.md`](doc/PYTHON_USAGE.md).

## Features

- Capability-based registry and stable aliases
- Explicit model availability and failure reasons
- Strict sequence canonicalization and amino-acid validation
- FASTA parsing
- Duplicate-ID and region-coordinate validation
- Mean, max, residue, region-mean, and window-mean representations
- Optional L2 normalization with original norms retained
- Cosine similarity and distance
- Angular, correlation, and Euclidean distance
- Sequence length ratio
- Global sequence identity
- 3-mer Jaccard similarity
- Normalized global and local BLOSUM62 scores
- Residue cosine-similarity matrices
- Needleman-Wunsch and Smith-Waterman embedding alignment
- Symmetric residue best-match score
- Balanced entropy-regularized Sinkhorn comparison
- Cross-model Spearman and Pearson relationship-matrix agreement
- Cross-model linear CKA across different embedding dimensions
- Nearest-neighbor overlap, rank consensus, and pair-disagreement analysis
- Single-protein residue CKA and representational-similarity analysis (RSA)
- Cross-comparison of completed jobs without rerunning PLMs
- Strict JSON import of precomputed pooled or residue embedding spaces
- Norm, duplicate, anisotropy, per-dimension variance, distance-concentration, and
  hubness diagnostics for pooled embedding spaces
- Mean-centering jobs that preserve source vectors, with explicit fit subsets,
  parameter checksums, and before/after diagnostics
- Categorical annotation analysis with KNN consistency, chance adjustment,
  per-class retrieval summaries, and class-mixing matrices
- Exact all-to-all safety threshold
- Partial failure isolation across models
- Atomic artifacts with SHA-256 checksums
- SVG heatmap, score distribution, nearest-neighbor graph, residue map, and
  model-agreement plot
- Optional Plotly PCoA, UMAP, and dendrogram
- FastAPI endpoint definitions and asynchronous comparison jobs
- Dependency-free deterministic adapter for tests and demonstrations

## Model registry

A registry entry identifies a registered adapter and checkpoint. It does not
guarantee that the model can load locally. The `availability` field reports
local backend checks; `conformance` describes the validation recorded for that
checkpoint.

| Framework ID | Family | Upstream artifact | Backend | Status |
|---|---|---|---|---|
| `reference-deterministic-v1` | Test reference | Built in | Standard library | Tested; not a biological PLM |
| `esm2-8m` | ESM-2 | `facebook/esm2_t6_8M_UR50D` | Hugging Face | CPU shapes, masking, and batch invariance tested at pinned revision |
| `esm2-35m` | ESM-2 | `facebook/esm2_t12_35M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-150m` | ESM-2 | `facebook/esm2_t30_150M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-650m` | ESM-2 | `facebook/esm2_t33_650M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-3b` | ESM-2 | `facebook/esm2_t36_3B_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-15b` | ESM-2 | `facebook/esm2_t48_15B_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm1b-650m` | ESM-1b | `facebook/esm1b_t33_650M_UR50S` | Hugging Face | Revision pinned; checkpoint validation required |
| `esm1v-650m-1` | ESM-1v | `facebook/esm1v_t33_650M_UR90S_1` | Hugging Face | Revision pinned; checkpoint validation required |
| `esm1v-650m-2` | ESM-1v | `facebook/esm1v_t33_650M_UR90S_2` | Hugging Face | Revision pinned; checkpoint validation required |
| `esm1v-650m-3` | ESM-1v | `facebook/esm1v_t33_650M_UR90S_3` | Hugging Face | Revision pinned; checkpoint validation required |
| `esm1v-650m-4` | ESM-1v | `facebook/esm1v_t33_650M_UR90S_4` | Hugging Face | Revision pinned; checkpoint validation required |
| `esm1v-650m-5` | ESM-1v | `facebook/esm1v_t33_650M_UR90S_5` | Hugging Face | Revision pinned; checkpoint validation required |
| `esmc-300m` | ESM-C | `biohub/ESMC-300M` | Biohub `esm` | 960 dimensions, 2,046 residues; checkpoint validation required |
| `esmc-600m` | ESM-C | `biohub/ESMC-600M` | Biohub `esm` | 1,152 dimensions, 2,046 residues; checkpoint validation required |
| `esmc-6b` | ESM-C | `biohub/ESMC-6B` | Biohub `esm` | 2,560 dimensions, 2,046 residues; very large checkpoint |
| `prott5-xl-uniref50` | ProtTrans | `Rostlab/prot_t5_xl_uniref50` | Hugging Face | Adapter implemented; large checkpoint |
| `prott5-xl-bfd` | ProtTrans | `Rostlab/prot_t5_xl_bfd` | Hugging Face | Revision pinned; large checkpoint; validation required |
| `protbert` | ProtTrans | `Rostlab/prot_bert` | Hugging Face | Revision pinned; validation required |
| `protbert-bfd` | ProtTrans | `Rostlab/prot_bert_bfd` | Hugging Face | Adapter implemented; validation required |
| `protalbert` | ProtTrans | `Rostlab/prot_albert` | Hugging Face | Revision pinned; validation required |
| `protxlnet` | ProtTrans | `Rostlab/prot_xlnet` | Hugging Face | Revision pinned; validation required |
| `ankh-base` | Ankh | `ElnaggarLab/ankh-base` | Hugging Face | Non-commercial license; validation required |
| `ankh-large` | Ankh | `ElnaggarLab/ankh-large` | Hugging Face | Non-commercial license; validation required |
| `progen2-small` | ProGen2 | `hugohrban/progen2-small` | Hugging Face custom code | Pinned community conversion; validation required |
| `progen2-medium` | ProGen2 | `hugohrban/progen2-medium` | Hugging Face custom code | Pinned community conversion; validation required |

Always inspect `GET /v1/models/{model_id}` before using a model. The registry
returns its current license note, limits, supported methods, and local
availability. Users remain responsible for complying with upstream terms.

## Installation

The core library requires Python 3.11 or newer. The complete installation,
including ESM-C, requires Python 3.12 or newer.

### Complete one-file installation

From the repository root, create a Python 3.12+ environment and install the
package with every runtime backend from `requirements.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the API, Hugging Face, ESM-C, and advanced-visualization
dependencies. Model checkpoints are still downloaded only when a real model is
first loaded. The dependency declarations live in `pyproject.toml`; the
requirements file selects the `all` installation profile to avoid duplicating
those declarations.

### Selective and development installation

For a smaller editable installation, install only the profiles needed for the
current environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e '.[dev]'
```

### API

```bash
python -m pip install -e '.[api]'
```

### Hugging Face models

```bash
python -m pip install -e '.[hf]'
```

### ESM-C models

ESM-C uses the maintained Biohub `esm` backend and currently requires Python
3.12 or newer:

```bash
python -m pip install -e '.[esmc]'
```

It uses a separate adapter from the Hugging Face models. See the
[Python guide](doc/PYTHON_USAGE.md#esm-c-setup-and-troubleshooting) for a strict
smoke test, supported devices/dtypes, and common failure messages.

### Advanced visualizations

```bash
python -m pip install -e '.[visual]'
```

The equivalent editable installation of all runtime backends plus development
tools is:

```bash
python -m pip install -e '.[all,dev]'
```

Large checkpoints are never downloaded during normal unit tests. Download size,
GPU memory, model license, and gated access should be reviewed before enabling a
real model in production.

## Workflows and flags

Run these commands from the repository root after installing the package.
The deterministic adapter is a software fixture and needs no checkpoint.

| Goal | Command | Flags and details |
|---|---|---|
| Compare two raw sequences | `python examples/test_two_seqs.py` | [Pair example and flags](doc/WORKFLOWS.md#two-sequence-flags) |
| Score reference/noisy FASTAs | `python examples/noisy_seq_analyser.py examples/noisy_reference.fasta examples/noisy_fastas --models reference-deterministic-v1` | [Analyser workflow and all flags](doc/NOISY_SEQUENCE_ANALYSER.md) |
| Inspect models, estimate, embed, compare | `plms-framework models`, `estimate`, `embed`, `compare` | [CLI commands](doc/WORKFLOWS.md#run-workflows) and [request options](doc/PYTHON_USAGE.md#complete-request-option-reference) |
| Use the library directly | `python examples/python_ex.py` | [Python demonstration flags](doc/WORKFLOWS.md#python-demonstration-flags) and [Python guide](doc/PYTHON_USAGE.md) |
| Run the API | `plms-framework serve` | [Server flags](doc/WORKFLOWS.md#server-flags) and [API endpoints](#api-endpoints) |
| Check the project | `python -m pytest -q` | [Automated workflows](doc/WORKFLOWS.md#automated-checks) |

The [two-sequence example](examples/test_two_seqs.py) accepts two raw protein
sequences and calls `score_two_sequences()`. That scorer uses the same sequence,
pooled-embedding, and residue metrics as `examples/noisy_seq_analyser.py`:

```bash
python examples/test_two_seqs.py
python examples/test_two_seqs.py MKTAYIAK MKTAYIAQ \
  --models esm2-8m --device cpu --precision float32
```

The first command uses `reference-deterministic-v1` and prints a JSON dictionary
with 19 scores: five sequence scores, ten pooled-vector scores, and four residue
scores. Two fields from the bundled example are:

| Score | Value |
|---|---:|
| `sequence.sequence_identity` | `0.875` |
| `reference-deterministic-v1.mean.cosine_similarity` | about `0.8913` |

The second command uses a real ESM-2 checkpoint; the first run may download its
weights. The [`score_two_sequences.py`](examples/score_two_sequences.py) CLI defaults to
`esm2-8m`.

For a FASTA directory, the included fixtures produce three matched pairs:

```bash
python examples/noisy_seq_analyser.py examples/noisy_reference.fasta examples/noisy_fastas \
  --models reference-deterministic-v1 --output noisy_sequence_scores.csv
```

Use `--model-wise --output model_scores` for one CSV per model. The analyser's
default model list includes large ESM checkpoints, so set `--models` explicitly
for a first run. See the [flag table](doc/NOISY_SEQUENCE_ANALYSER.md#flags) for
matching, workers, batching, cache, length limits, and logging.

## Dependency-free example

The built-in reference adapter is a deterministic software test fixture, not a
biological PLM:

```bash
PYTHONPATH=src python -m plms_framework.cli compare examples/compare.json
```

To list registered models and local availability:

```bash
PYTHONPATH=src python -m plms_framework.cli models
```

To estimate storage before inference:

```bash
PYTHONPATH=src python -m plms_framework.cli estimate examples/embed.json
```

## Start the API

```bash
plms-framework serve --host 127.0.0.1 --port 8000
```

Or:

```bash
uvicorn plms_framework.api:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

The default artifact directory is `.plms_artifacts`. Configure it without
placing secrets in the path:

```bash
export PLMS_ARTIFACT_DIR=/srv/plms/artifacts
export PLMS_JOB_WORKERS=1
```

One worker is conservative because loading several large checkpoints at once can
exhaust GPU memory.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API process health |
| `GET` | `/ready` | Registry and locally available model count |
| `GET` | `/v1/models` | Discover manifests and availability |
| `GET` | `/v1/models/{id}` | Inspect one model |
| `POST` | `/v1/estimate` | Estimate residues, pairs, matrix bytes, and embedding bytes |
| `POST` | `/v1/embed` | Synchronous embedding request |
| `POST` | `/v1/spaces/import` | Import one precomputed embedding space as an asynchronous job |
| `POST` | `/v1/diagnostics` | Diagnose pooled embedding geometry from a successful job |
| `POST` | `/v1/transform` | Create a provenance-tracked embedding transformation from a successful job |
| `POST` | `/v1/annotations/analyze` | Measure categorical-label organization in embedding neighborhoods |
| `POST` | `/v1/compare` | Submit asynchronous all-to-all comparison |
| `POST` | `/v1/cross-compare` | Compare models from completed jobs without rerunning inference |
| `POST` | `/v1/benchmark` | Submit efficiency benchmark; biological tracks require datasets |
| `POST` | `/v1/fasta/parse` | Validate and convert FASTA upload |
| `GET` | `/v1/jobs/{id}` | Poll job status; optionally include result |
| `GET` | `/v1/jobs/{id}/artifacts` | List checksummed artifacts |
| `GET` | `/v1/jobs/{id}/artifacts/{name}` | Download one artifact |

## Request format

This example is a comparison request for `POST /v1/compare` or the CLI
`compare` command. It uses the `hf` extra for ESM-2 and the `visual` extra for
PCoA. For `POST /v1/embed`, omit the comparison-only `metrics`,
`visualizations`, and `cross_model` fields.

```json
{
  "models": ["esm2-8m"],
  "sequences": [
    {
      "id": "protein_a",
      "sequence": "MKTAYIAKQRQISFVKSHFSRQ",
      "metadata": {"family": "demo", "organism": "example"},
      "regions": [{"name": "n_terminal", "start": 1, "end": 10}]
    },
    {
      "id": "protein_b",
      "sequence": "MKTAYIAKQKQISFVKSHFSRQ",
      "regions": [{"name": "n_terminal", "start": 1, "end": 10}]
    }
  ],
  "representations": [
    {"method": "mean", "layer": "model_default", "normalize": "l2"},
    {"method": "residue", "layer": "model_default", "normalize": "none"},
    {"method": "region_mean", "normalize": "l2"}
  ],
  "metrics": [
    "cosine_similarity",
    "angular_distance",
    "sequence_identity",
    "blosum62_global",
    "residue_local_alignment",
    "residue_optimal_transport"
  ],
  "visualizations": [
    "clustered_heatmap",
    "pcoa",
    "residue_similarity_map",
    "distance_distribution"
  ],
  "cross_model": {
    "enabled": false,
    "methods": ["matrix_spearman"],
    "representation": "mean",
    "neighbor_k": [1, 5, 10],
    "max_disagreements": 20
  },
  "execution": {
    "precision": "auto",
    "device": "auto",
    "all_to_all": "exact",
    "on_long_sequence": "error",
    "allow_partial": true,
    "max_exact_sequences": 2000
  }
}
```

### Models

Use registered framework IDs or aliases. Several models may be selected in
one request. A model failure is isolated when `allow_partial` is true and
appears in the result's `failures` list. Set `allow_partial` to false to raise
the first model failure.

### Sequences

Sequences are uppercased and formatting whitespace is removed. Identifiers
are stripped of leading and trailing whitespace and must then be non-empty
and unique. Regions use **1-based inclusive** coordinates within the sequence.
Characters outside the model's declared alphabet produce a validation error.

### Representation methods

| Method | Shape per sequence | Meaning |
|---|---:|---|
| `mean` | `D` | Mean of valid residue vectors; default |
| `residue` | `L × D` | Contextual vector for each native residue |
| `native` | `D` | Upstream-defined sequence vector; only on adapters that justify it |
| `max` | `D` | Feature-wise residue maximum; experimental diagnostic |
| `region_mean` | `R × D` | Mean for each submitted region |
| `window_mean` | `W × D` | Mean for deterministic windows |

`region_mean` requires at least one region on every sequence, as shown in
the example. For `window_mean`, provide a positive `window_size` and optionally
a positive `window_step`; the step defaults to the size. The last partial
window is retained. These windows pool existing residue embeddings and do not
extend the model's context length.

`normalize` is `l2` (the default) or `none`; residue embeddings are normalized
row by row. No default registry model currently supports `native` output.
Region and window values include coordinates alongside their vectors.

Layer is independent of pooling. Use `model_default` unless a model-specific
analysis justifies another layer. An integer layer is validated against the
manifest when layer count is known.

### Vector metrics

| Metric | Interpretation |
|---|---|
| `cosine_similarity` | Primary global similarity; higher is more similar |
| `cosine_distance` | `1 - cosine`; lower is closer |
| `angular_distance` | `acos(cosine) / π`; angular separation scaled to `[0, 1]` |
| `correlation_distance` | Cosine distance after feature centering |
| `euclidean_distance` | Geometric distance; scale-sensitive without normalization |

Cosine similarity is reported as zero if either vector has zero norm.

For nonzero L2-normalized vectors, squared Euclidean distance equals
`2 - 2 × cosine_similarity`; these are not independent measurements.

### Sequence controls

| Metric | Current definition |
|---|---|
| `length_ratio` | Shorter length divided by longer length |
| `sequence_identity` | Needleman-Wunsch, match `1`, mismatch `-1`, gap `-1`; matches/alignment columns |
| `kmer_jaccard` | Jaccard similarity of unique 3-mers |
| `blosum62_global` | Global BLOSUM62, linear gap `-4`, geometric self-score normalization |
| `blosum62_local` | Local BLOSUM62, linear gap `-4`, geometric self-score normalization |

The result includes `metric_definitions` so downstream users do not need to
infer these choices.

### Residue metrics

Residue metrics require exactly one `residue` method/layer specification in
the same request:

- `residue_global_alignment`: Needleman-Wunsch over residue cosine scores.
- `residue_local_alignment`: Smith-Waterman over residue cosine scores.
- `residue_best_match`: average of bidirectional best residue matches.
- `residue_optimal_transport`: balanced entropy-regularized Sinkhorn transport
  with uniform residue mass.

Global and local embedding alignment currently use a linear `-0.5` gap penalty.
The output retains the raw score, path-length-normalized score, mean matched
similarity, and 1-based alignment path. Sinkhorn is balanced OT, not yet the
gap-aware unbalanced method planned for later.

### Visualizations

| Name | Output | Notes |
|---|---|---|
| `clustered_heatmap` | SVG | Deterministic average-linkage ordering |
| `distance_distribution` | SVG | Upper-triangle score distribution |
| `nearest_neighbor_graph` | SVG | Top-3 exploratory graph |
| `residue_similarity_map` | SVG per model and pair | Requires residue comparison data |
| `model_agreement` | SVG | Spearman agreement across models |
| `model_matrix_small_multiples` | SVG per model | Identically ordered relationship views |
| `consensus_heatmap` | SVG | Rank-normalized multi-model consensus |
| `model_disagreement` | SVG | Pairs whose relative ranks differ across models |
| `residue_model_agreement` | SVG per sequence | Model-by-model residue CKA or RSA |
| `pcoa` | Interactive HTML | Requires `visual` extra |
| `umap` | Interactive HTML | Exploratory; seed fixed; requires `visual` extra |
| `dendrogram` | Interactive HTML | Average linkage; requires `visual` extra |

Heatmaps, score distributions, neighbor graphs, PCoA, UMAP, and dendrograms
use the first available model metric matrix, falling back to the first sequence
baseline if no model matrix is available. Model agreement plots need the
corresponding enabled cross-model methods.

UMAP is exploratory and can exaggerate apparent gaps between clusters. Inspect
the original pairwise matrices when interpreting any projection.

## Single-sequence behavior

`POST /v1/embed` returns mean pooling by default. Multiple representations may be
requested. Result records contain:

- Sequence ID
- Method and layer
- Values and numerical shape
- Norm before optional normalization
- Actual normalization
- Adapter and execution metadata

Residue outputs may be large. The current synchronous endpoint includes them in
JSON; binary externalization is a priority before exposing the service to long
proteins or large batches.

## Importing precomputed embedding spaces

`POST /v1/spaces/import` makes embeddings generated outside this framework
available to `/v1/cross-compare`. The initial import format is JSON so its
validation and provenance behavior remain dependency-free and transparent.
Compressed NPZ, Parquet, and Zarr imports and exports are not yet implemented.

One import job represents one named embedding space and may contain pooled and
residue representations:

```json
{
  "space_name": "external-esm-run",
  "source": {
    "display_name": "ESM experiment from another cluster",
    "family": "ESM-2",
    "provider": "example-lab",
    "artifact": "facebook/esm2_t33_650M_UR50D",
    "revision": "immutable-commit-or-unreported",
    "license": "record-the-upstream-license",
    "citation": "record-the-model-citation"
  },
  "representations": [
    {
      "method": "mean",
      "layer": 33,
      "normalization": "none",
      "embeddings": [
        {"sequence_id": "protein_a", "values": [0.1, 0.2, -0.3]},
        {"sequence_id": "protein_b", "values": [0.4, -0.1, 0.2]},
        {"sequence_id": "protein_c", "values": [-0.2, 0.3, 0.1]}
      ]
    }
  ]
}
```

These three-dimensional vectors are synthetic examples, not ESM model output.
Replace the values and source metadata with those from your own run.

The complete request is in `examples/import_space.json`. Poll the returned job
ID through `/v1/jobs/{id}`. The successful import job can then be included in
the `sources` list of `/v1/cross-compare`, alongside other imports or ordinary
framework comparison jobs.

Current JSON imports support `mean`, `max`, `native`, and `residue`. Validation
requires:

- Unique, non-empty sequence IDs
- Identical IDs and ordering across every representation in the import
- A consistent feature dimension within each representation
- Rectangular residue matrices
- Numeric, finite values—NaN and infinity are rejected
- At most 10,000 records per representation, 32,768 features per vector, and
  5,000,000 scalar values across the import
- A unique method/layer combination inside one imported space

`normalization` is a declaration about the supplied data, not a transformation.
Allowed declarations are `unknown`, `none`, `l2`, `mean_centered`,
`standardized`, `whitened`, and `custom`.

The framework computes a deterministic SHA-256 digest over the import payload
and records source fields in provenance. It does **not** claim that imported
values came from the named checkpoint. Imported manifests therefore carry
`externally_supplied_structural_validation_only` conformance. Clients should
provide immutable revisions and their own source checksums whenever available.

## Embedding-space diagnostics

`POST /v1/diagnostics` analyzes pooled vectors from a successful import or
comparison job without modifying them:

```json
{
  "source_job_id": "SUCCESSFUL_JOB_ID",
  "representation": "mean",
  "layer": null,
  "sample_pairs": 10000,
  "hubness_k": 10,
  "seed": 42
}
```

The complete payload is in `examples/diagnostics.json`. The endpoint creates an
asynchronous job and reports each source model separately. When `layer` is null,
the first complete matching layer is selected and a warning is emitted if more
than one is available.

Current diagnostics include:

- Vector norm minimum, mean, maximum, and zero-vector count
- Exact duplicate vectors and affected sequence IDs
- Per-dimension means and population variances
- A variance-based participation ratio: `(sum(v))² / sum(v²)`, where `v`
  contains the per-dimension variances; this is not a covariance-spectrum
  estimate of intrinsic dimension
- Average raw and mean-centered cosine over reproducibly sampled sequence pairs
- An anisotropy score defined as average cosine divided by `1/sqrt(D)`
- Sampled cosine-distance concentration statistics
- Exact top-k occurrence, top hubs, and Robin Hood hubness index

The same sampled pair indices are used for every model in a job. All-pair values
are used when the collection has no more than `sample_pairs` distinct pairs.
Exact hubness is limited to 2,000 sequences; above that size the hubness result
is explicitly unavailable until an approximate-neighbor backend is implemented.

These measurements characterize geometry, not biological performance. In
particular, a low anisotropy or hubness score does not prove that an embedding
captures protein function, structure, or evolutionary relationships.

## Mean-centering

`POST /v1/transform` creates a new embedding-space job without modifying its
source. Mean-centering is currently available for `mean`, `max`, and `native`
vectors:

```json
{
  "source_job_id": "SUCCESSFUL_JOB_ID",
  "method": "mean_center",
  "representation": "mean",
  "layer": null,
  "fit_sequence_ids": ["training_a", "training_b"],
  "sample_pairs": 10000,
  "hubness_k": 10,
  "seed": 42
}
```

The fitted mean is calculated independently for each model. Every transformed
space receives a derived identifier such as `esm2-8m::mean-centered`. The result
contains:

- The transformed vectors marked `mean_centered`, with the source
  normalization retained in each record's metadata
- The IDs used to fit the mean
- The fitted mean vector and its SHA-256 checksum
- Geometry diagnostics before and after transformation
- A source-to-derived model mapping
- A warning when the mean was fitted on all available sequences

The job also writes `transformation_parameters.json`. Adding the retained mean
vector to a transformed vector reconstructs the source vector, subject to normal
floating-point precision.

For exploratory analysis, omitting `fit_sequence_ids` fits the mean on the whole
source collection. For held-out evaluation, provide only training or reference
IDs. Fitting on test proteins would leak information from the evaluation set
into the transformation.

The complete request is in `examples/transform_mean_center.json`. Transformation
jobs can be supplied directly to `/v1/diagnostics` or `/v1/cross-compare`.

## Categorical annotation analysis

`POST /v1/annotations/analyze` measures how supplied categorical labels are
organized within each model's embedding neighborhoods:

```json
{
  "source_job_id": "SUCCESSFUL_JOB_ID",
  "feature": "protein_family",
  "representation": "mean",
  "layer": null,
  "k": [1, 5, 10],
  "missing_policy": "error",
  "annotations": [
    {
      "sequence_id": "protein_a",
      "categorical": {"protein_family": "kinase"},
      "numerical": {"length": 287}
    },
    {
      "sequence_id": "protein_b",
      "categorical": {"protein_family": "phosphatase"},
      "numerical": {"length": 301}
    }
  ]
}
```

The analysis currently uses one categorical feature and within-model cosine
neighbors. Categorical values may be strings, integers, or booleans. A null
value is treated as a missing label and follows `missing_policy`. Numerical
annotations are accepted by the typed request schema but are not analyzed or
copied into the result; keep the original annotations if you need them later.

For every requested valid `k`, each model reports:

- Per-sequence neighbor IDs, label consistency, and same-class recall
- Micro- and macro-averaged consistency
- The expected same-label rate under the observed class distribution
- Chance-adjusted consistency
- Per-class consistency and recall
- Class-by-class neighbor counts and row-normalized mixing fractions

The response includes class sizes, dropped sequence IDs, warnings for singleton
or small classes, and a SHA-256 digest of the labels used. `missing_policy` is
`error` by default; set it to `drop` to analyze only labeled source sequences.
Exact analysis is limited to 2,000 labeled sequences.

These are descriptive in-sample neighborhood measurements. They do not replace
family-aware train/test evaluation and do not establish that the labels are
correct. The complete request is in `examples/annotation_analysis.json`.

## Multi-sequence behavior

`POST /v1/compare` returns `202 Accepted` with a job ID. Poll:

```bash
curl 'http://127.0.0.1:8000/v1/jobs/JOB_ID?include_result=true'
```

The result has stable `sequence_order`, per-model representations and matrices,
sequence baselines, residue-pair analyses, model agreement, failures, and run
provenance. Artifacts are written atomically and accompanied by SHA-256 hashes.

## Cross-model comparison

Suppose the same proteins are embedded by ESM-2 and ProtT5. Even if their vectors
had the same dimension, coordinate 17 in one model has no shared definition with
coordinate 17 in the other. Direct vector cosine is therefore invalid.

The framework instead builds an `N × N` similarity matrix independently for each
model. It compares the resulting relationships. This asks:

> Do the models rank protein relationships similarly?

Enable it directly in `/v1/compare`:

```json
{
  "models": ["esm2-8m", "esmc-300m", "prott5-xl-uniref50"],
  "sequences": [
    {"id": "a", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"},
    {"id": "b", "sequence": "MKTAYIAKQKQISFVKSHFSRQ"},
    {"id": "c", "sequence": "GGGGGGGGGPGGGGGGGGGG"}
  ],
  "representations": [{"method": "mean", "normalize": "l2"}],
  "metrics": ["cosine_similarity"],
  "cross_model": {
    "enabled": true,
    "representation": "mean",
    "methods": [
      "matrix_spearman",
      "matrix_pearson",
      "linear_cka",
      "neighbor_overlap",
      "consensus_rank",
      "pair_disagreement"
    ],
    "neighbor_k": [1, 2],
    "max_disagreements": 20
  },
  "visualizations": [
    "model_agreement",
    "model_matrix_small_multiples",
    "consensus_heatmap",
    "model_disagreement"
  ]
}
```

### Cross-model methods for multiple sequences

| Method | Shared comparison object | Minimum |
|---|---|---:|
| `matrix_spearman` | Rank correlation of upper-triangle cosine relationships | 3 sequences |
| `matrix_pearson` | Linear correlation of upper-triangle cosine relationships | 3 sequences |
| `linear_cka` | Centered alignment of `N × D` representation matrices | 2 sequences; more strongly recommended |
| `neighbor_overlap` | Mean Jaccard overlap of top-k neighbors | 2 sequences |
| `consensus_rank` | Mean within-model pair-rank percentile | 3 sequences |
| `pair_disagreement` | Range of pair-rank percentiles across models | 3 sequences |

Matrix correlations with two sequences are undefined because their upper
triangle contains only one observation. The API returns an explicit unavailable
result rather than a misleading zero. CKA with two sequences is calculable but
usually uninformative; the result includes a warning. For scientific analysis,
use at least 20–50 diverse sequences where possible.

Consensus and disagreement use **within-model ranks**, not averages of raw cosine
values. This prevents models with different cosine ranges from dominating the
result.

### One sequence across multiple PLMs

A single pooled vector from each PLM cannot support valid cross-model comparison.
Request residue embeddings instead:

```json
{
  "models": ["esm2-8m", "esmc-300m"],
  "sequences": [{"id": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"}],
  "representations": [{"method": "residue", "normalize": "none"}],
  "metrics": [],
  "cross_model": {
    "enabled": true,
    "methods": ["residue_cka", "residue_rsa"]
  },
  "visualizations": ["residue_model_agreement"]
}
```

- `residue_cka` treats aligned residues as shared observations and permits
  different feature dimensions.
- `residue_rsa` creates an `L × L` residue self-similarity matrix for each model
  and correlates the upper triangles.

Both require consistent token-to-residue mapping. No individual ESM coordinate
is compared to a ProtT5 or ESM-C coordinate.

### Reusing existing jobs

`POST /v1/cross-compare` accepts one or more successful comparison, import, or
transformation job IDs containing at least two models in total. The synchronous
`/v1/embed` endpoint does not create a job ID. All sources must contain identical
sequence IDs in identical order, and a model may occur only once:

```json
{
  "sources": [
    {"job_id": "JOB_FROM_ESM2"},
    {"job_id": "JOB_FROM_PROTT5"}
  ],
  "representation": "mean",
  "methods": ["matrix_spearman", "linear_cka", "consensus_rank"],
  "neighbor_k": [1, 5, 10],
  "max_disagreements": 20,
  "visualizations": ["model_agreement", "consensus_heatmap"]
}
```

This endpoint creates another asynchronous job but does not rerun either PLM.
It is useful when embeddings were produced at different times or on different
hardware. The source jobs must still be available in the current job store;
persistent cross-deployment artifact import is planned with persistent workers.

Cross-model output includes:

- Successful model order and models with complete relationship matrices
- Per-model within-model cosine matrices
- Requested model-by-model agreement matrices
- Neighbor overlap at every valid requested `k`
- Consensus sequence matrix
- Highest-disagreement sequence pairs and per-model rank percentiles
- Per-sequence residue agreement matrices
- Warnings for inadequate sample sizes or missing representations
- The explicit guardrail: raw cross-PLM vectors were not directly compared

## Long sequences

Sequences are never silently truncated. When a sequence exceeds a model's
declared native limit, the current behavior is an error naming the model and
sequence IDs.

Choose a model with a suitable declared length limit. For windowed inference,
the planned steps are:

- Apply named overlapping windows.
- Merge residue vectors with recorded edge weighting.
- Mark the result as non-native and warn that global context was lost.

`on_long_sequence="error"` is currently the only supported policy.
`window_mean` pools an existing residue embedding; it does not extend a model's
context length.

## Exact all-to-all scaling

For `N` sequences, each metric contains `N²` entries and evaluates
`N(N-1)/2` distinct pairs. A float64 matrix needs approximately `8N²` bytes:

| N | One matrix | Distinct pairs |
|---:|---:|---:|
| 1,000 | 8 MB | 499,500 |
| 10,000 | 800 MB | 49,995,000 |
| 100,000 | 80 GB | 4,999,950,000 |

The default exact limit is 2,000 sequences and is configurable per request.
Raising it increases compute and storage costs. These byte counts describe
packed float64 arrays; the current Python lists and JSON output use more memory
and disk space. Approximate nearest-neighbor mode is planned but is never substituted silently.

Residue all-to-all comparison is more expensive: a pair of lengths `L1` and
`L2` needs an `L1 × L2` similarity matrix. Request it only when biologically
needed.

## Provenance

Each run currently records:

- Framework and schema version
- Python and platform
- Embedding start, finish, and duration; comparison metrics run after this
  interval, so it is not the total comparison time
- Sequence SHA-256 digests and lengths
- Execution choices
- Per-model artifact, requested revision, backend, and adapter version
- Representation method, layer, shape, normalization, and original norm
- Metric definitions
- Partial failures
- Artifact names, sizes, media types, and checksums

ESM-2 8M, ESM-1b/1v, several ProtTrans models, and the ProGen2 conversions
have pinned revisions. Other manifests still request `main`. The Hugging Face
adapter records the resolved commit when the backend exposes it; the ESM-C
adapter currently records only the requested revision. To reproduce model
loading, pin revisions and keep checkpoint checksums alongside the results.

Complete sequence content is not written to ordinary logs. Deployments handling
unpublished sequences should also control result access, artifact retention,
backups, and infrastructure logs.

## Adding a model adapter

1. Create a `ModelManifest` with an immutable upstream revision when possible.
2. Subclass `ModelAdapter`.
3. Implement `availability()` without downloading the checkpoint.
4. Implement lazy `load()` and safe `unload()`.
5. Validate model-specific sequence length, alphabet, layer, and precision.
6. Return one `RawEmbedding` per sequence with exactly `L` residue rows.
   Validate that each retained token corresponds to the expected amino acid;
   a matching row count alone does not establish correct tokenization.
7. Return an explicit token-to-residue map.
8. Add native sequence output only when scientifically justified.
9. Add reference-output, masking, batch-invariance, and determinism tests.
10. Mark conformance as tested only after real checkpoint validation.

The core takes responsibility for deterministic pooling and metrics. An adapter
should not secretly pool, normalize, truncate, or replace residues.

Minimal shape:

```python
class MyAdapter(ModelAdapter):
    def availability(self) -> Availability:
        ...

    def embed(self, sequences, *, layer="model_default", device="auto", precision="auto"):
        # Return len(sequence) residue rows for every input.
        return [RawEmbedding(...)]
```

Register it with a factory rather than a loaded model instance so registry
discovery remains cheap.
