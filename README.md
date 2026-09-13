# PLMs Framework

PLMs Framework is a capability-driven API and Python library for extracting,
comparing, benchmarking, and visualizing representations from protein language
models.

Its purpose is broader than wrapping one model family. The framework provides a
stable scientific workflow around models that differ in architecture,
tokenization, sequence length, representation dimension, licensing, and hardware
requirements.

> **Current maturity:** early working foundation. The real ESM-2 8M adapter and
> dependency-free reference path, model registry, pooling, sequence and embedding metrics, residue
> alignment, optimal transport, provenance, background jobs, artifact storage,
> API design, and tests are implemented. Other real checkpoint adapters are
> present but require their optional dependencies, checkpoint downloads, and
> model-level conformance validation before production claims should be made.

The complete product specification is in
[`IMPLEMENTATION_PROMPT.md`](IMPLEMENTATION_PROMPT.md).

## Why this framework exists

PLMs commonly expose contextual vectors for every amino acid. Turning those
vectors into a trustworthy protein comparison is not just a call to
`model.forward()`:

- Models use different tokenizers and boundary tokens.
- Padding and special tokens must not leak into protein pooling.
- Long sequences must never be silently truncated.
- Some models have a meaningful native sequence representation; others do not.
- Raw vectors from different models do **not** occupy a shared coordinate system.
- An attractive UMAP plot does not demonstrate biological performance.
- Model aliases, revisions, precision, pooling, and preprocessing must be retained
  to reproduce a result.

PLMs Framework makes these choices explicit and inspectable.

## Scientific contract

The framework follows five central rules:

1. **Mean pooling is the portable default.** It averages valid residue vectors,
   excluding padding and boundary tokens.
2. **Residue representations remain available.** They support motif discovery,
   local similarity, embedding-based alignment, and residue-level tasks.
3. **Raw vectors are compared only within the same model and representation.**
   ESM-2 and ProtT5 coordinates cannot be compared directly.
4. **Models are compared through relationships.** Pairwise matrices, rankings,
   neighbor overlap, biological tasks, and representational-similarity methods
   are valid cross-model objects.
5. **Nothing important is silent.** Truncation, unavailable checkpoints,
   approximation, partial failures, normalization, and adapter limitations are
   explicit.

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

## Implemented features

### Working now

- Capability-based registry and stable aliases
- Explicit model availability and failure reasons
- Strict sequence canonicalization and amino-acid validation
- FASTA parsing
- Duplicate-ID and region-coordinate validation
- Mean, max, residue, region-mean, and window-mean representations
- Optional L2 normalization with original norms retained
- Cosine similarity and distance
- Angular, correlation, and Euclidean distance
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
- Norm, duplicate, anisotropy, effective-dimension, distance-concentration, and
  hubness diagnostics for pooled embedding spaces
- Exact all-to-all safety threshold
- Partial failure isolation across models
- Atomic artifacts with SHA-256 checksums
- SVG heatmap, score distribution, nearest-neighbor graph, residue map, and
  model-agreement plot
- Optional Plotly PCoA, UMAP, and dendrogram
- FastAPI endpoint definitions and asynchronous comparison jobs
- Dependency-free deterministic adapter for tests and demonstrations

### Designed but not yet complete

- Immutable upstream revision resolution after checkpoint download
- Content-addressed embedding cache
- Chunked binary storage for very large residue arrays
- Approximate nearest-neighbor execution for large collections
- Unbalanced/gap-aware optimal transport
- Job cancellation, persistent queue, authentication, quotas, and artifact expiry
- Versioned biological benchmark datasets and trained downstream heads
- Full conformance runs for every real checkpoint and hardware/precision pair
- MSA, structure-token, multimodal, inverse-folding, causal, and antibody adapters

These distinctions matter: a registry entry means an adapter is known, while
`conformance: tested` means its numerical behavior was validated.

## Initial model registry

| Framework ID | Family | Upstream artifact | Backend | Status |
|---|---|---|---|---|
| `reference-deterministic-v1` | Test reference | Built in | Standard library | Tested; not a biological PLM |
| `esm2-8m` | ESM-2 | `facebook/esm2_t6_8M_UR50D` | Hugging Face | CPU shapes, masking, and batch invariance tested at pinned revision |
| `esm2-35m` | ESM-2 | `facebook/esm2_t12_35M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-150m` | ESM-2 | `facebook/esm2_t30_150M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-650m` | ESM-2 | `facebook/esm2_t33_650M_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-3b` | ESM-2 | `facebook/esm2_t36_3B_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esm2-15b` | ESM-2 | `facebook/esm2_t48_15B_UR50D` | Hugging Face | Adapter implemented; checkpoint validation required |
| `esmc-300m` | ESM-C | `Biohub/ESMC-300M` | Hugging Face | Access/license and checkpoint validation required |
| `esmc-600m` | ESM-C | `Biohub/ESMC-600M` | Hugging Face | Same |
| `esmc-6b` | ESM-C | `Biohub/ESMC-6B` | Hugging Face | Same |
| `prott5-xl-uniref50` | ProtTrans | `Rostlab/prot_t5_xl_uniref50` | Hugging Face | Adapter implemented; large checkpoint |
| `protbert-bfd` | ProtTrans | `Rostlab/prot_bert_bfd` | Hugging Face | Adapter implemented; validation required |
| `ankh-base` | Ankh | `ElnaggarLab/ankh-base` | Hugging Face | Non-commercial license; validation required |
| `ankh-large` | Ankh | `ElnaggarLab/ankh-large` | Hugging Face | Non-commercial license; validation required |

Always inspect `GET /v1/models/{model_id}` before using a model. The registry
returns its current license note, limits, supported methods, and local
availability. Users remain responsible for complying with upstream terms.

## Installation

Python 3.11 or newer is required.

### Core and development

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

### Advanced visualizations

```bash
python -m pip install -e '.[visual]'
```

Install everything only when the machine is intended to support every backend:

```bash
python -m pip install -e '.[all,dev]'
```

Large checkpoints are never downloaded during normal unit tests. Download size,
GPU memory, model license, and gated access should be reviewed before enabling a
real model in production.

## Five-minute dependency-free example

The built-in reference adapter exercises the complete software workflow without
pretending to be a biological model:

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
| `POST` | `/v1/compare` | Submit asynchronous all-to-all comparison |
| `POST` | `/v1/cross-compare` | Compare models from completed jobs without rerunning inference |
| `POST` | `/v1/benchmark` | Submit efficiency benchmark; biological tracks require datasets |
| `POST` | `/v1/fasta/parse` | Validate and convert FASTA upload |
| `GET` | `/v1/jobs/{id}` | Poll job status; optionally include result |
| `GET` | `/v1/jobs/{id}/artifacts` | List checksummed artifacts |
| `GET` | `/v1/jobs/{id}/artifacts/{name}` | Download one artifact |

## Request format

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
      "sequence": "MKTAYIAKQKQISFVKSHFSRQ"
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

Use stable framework IDs. Several models may be selected in one request. A model
failure is isolated when `allow_partial` is true. A future `all_compatible`
resolver will require resource filters and an explicit download policy; it must
never initiate unbounded checkpoint downloads.

### Sequences

Sequences are uppercased and formatting whitespace is removed. Identifiers must
be unique. Regions use **1-based inclusive** coordinates. Invalid or unsupported
amino acids produce a validation error associated with the relevant sequence.

### Representation methods

| Method | Shape per sequence | Meaning |
|---|---:|---|
| `mean` | `D` | Mean of valid residue vectors; default |
| `residue` | `L × D` | Contextual vector for each native residue |
| `native` | `D` | Upstream-defined sequence vector; only on adapters that justify it |
| `max` | `D` | Feature-wise residue maximum; experimental diagnostic |
| `region_mean` | `R × D` | Mean for each submitted region |
| `window_mean` | `W × D` | Mean for deterministic windows |

For `window_mean`, provide positive `window_size` and optionally `window_step`.
The last partial window is retained. `normalize` is `l2` or `none`.

Layer is independent of pooling. Use `model_default` unless a model-specific
analysis justifies another layer. An integer layer is validated against the
manifest when layer count is known.

### Vector metrics

| Metric | Interpretation |
|---|---|
| `cosine_similarity` | Primary global similarity; higher is more similar |
| `cosine_distance` | `1 - cosine`; lower is closer |
| `angular_distance` | `acos(cosine) / π`; proper angular metric |
| `correlation_distance` | Cosine distance after feature centering |
| `euclidean_distance` | Geometric distance; scale-sensitive without normalization |

For L2-normalized vectors, squared Euclidean distance equals
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

Residue metrics require a `residue` representation in the same request:

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
| `residue_similarity_map` | SVG per pair | Requires residue comparison data |
| `model_agreement` | SVG | Spearman agreement across models |
| `model_matrix_small_multiples` | SVG per model | Identically ordered relationship views |
| `consensus_heatmap` | SVG | Rank-normalized multi-model consensus |
| `model_disagreement` | SVG | Pairs whose relative ranks differ across models |
| `residue_model_agreement` | SVG per sequence | Model-by-model residue CKA or RSA |
| `pcoa` | Interactive HTML | Requires `visual` extra |
| `umap` | Interactive HTML | Exploratory; seed fixed; requires `visual` extra |
| `dendrogram` | Interactive HTML | Average linkage; requires `visual` extra |

UMAP changes geometry and may exaggerate apparent gaps between clusters. Use the
matrix heatmap and PCoA when quantitative distances matter.

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
Compressed NPZ, Parquet, and Zarr transports are tracked in `ROADMAP.md`.

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
      "normalization": "l2",
      "embeddings": [
        {"sequence_id": "protein_a", "values": [0.1, 0.2, -0.3]},
        {"sequence_id": "protein_b", "values": [0.4, -0.1, 0.2]},
        {"sequence_id": "protein_c", "values": [-0.2, 0.3, 0.1]}
      ]
    }
  ]
}
```

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
- At most 10,000 records, 32,768 features, and 5,000,000 scalar values per job
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
- Participation ratio as an effective-dimension diagnostic
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

`POST /v1/cross-compare` accepts one or more successful embedding/comparison job
IDs containing at least two models in total. All sources must contain identical
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

Silent truncation is forbidden. When a sequence exceeds a model's declared native
limit, the current behavior is an error naming the model and sequence IDs.

Planned explicit alternatives are:

- Choose a compatible long-context model.
- Apply named overlapping windows.
- Merge residue vectors with recorded edge weighting.
- Mark the result as non-native and warn that global context was lost.

Until that policy is implemented and tested, `on_long_sequence` should remain
`error`.

## Exact all-to-all scaling

For `N` sequences, each metric contains `N²` entries and evaluates
`N(N-1)/2` distinct pairs. A float64 matrix needs approximately `8N²` bytes:

| N | One matrix | Distinct pairs |
|---:|---:|---:|
| 1,000 | 8 MB | 499,500 |
| 10,000 | 800 MB | 49,995,000 |
| 100,000 | 80 GB | 4,999,950,000 |

The default exact limit is 2,000 sequences and is configurable per request.
Raising it is an explicit acceptance of compute and storage cost. Approximate
nearest-neighbor mode is planned but is never substituted silently.

Residue all-to-all comparison is more expensive: a pair of lengths `L1` and
`L2` needs an `L1 × L2` similarity matrix. Request it only when biologically
needed.

## Provenance

Each run currently records:

- Framework and schema version
- Python and platform
- Start, finish, and duration
- Sequence SHA-256 digests and lengths
- Execution choices
- Per-model artifact, requested revision, backend, and adapter version
- Representation method, layer, shape, normalization, and original norm
- Metric definitions
- Partial failures
- Artifact names, sizes, media types, and checksums

The validated ESM-2 8M manifest is pinned to an immutable upstream revision, and
the Hugging Face adapter records its resolved commit. Other upstream aliases
currently resolve to their manifest's requested revision. A production
deployment should pin and checksum each checkpoint before advertising replay
guarantees for it.

Complete sequence content is not written to ordinary logs. Deployments handling
unpublished sequences should also control result access, artifact retention,
backups, and infrastructure logs.

## Benchmarking

`/v1/benchmark` currently implements an efficiency track over repeated embedding
runs and reports timing plus storage estimates. Biological benchmark tracks are
deliberately not faked: they return `not_configured` until a versioned dataset
manifest is supplied.

The planned tracks are:

- Frozen standardized representation
- Native model best practice
- Native zero-shot capability
- Efficiency and operational cost
- Learned pooling, reported separately

Biological evaluation should cover remote homology, function, localization,
interactions, variant effects, secondary structure, domains, and retrieval. Use
family-, identity-, temporal-, and pretraining-aware splits. Random protein
splits can leak close homologues and substantially overstate generalization.

## Adding a model adapter

1. Create a `ModelManifest` with an immutable upstream revision when possible.
2. Subclass `ModelAdapter`.
3. Implement `availability()` without downloading the checkpoint.
4. Implement lazy `load()` and safe `unload()`.
5. Validate model-specific sequence length, alphabet, layer, and precision.
6. Return one `RawEmbedding` per sequence with exactly `L` residue rows.
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

## Testing

The core suite needs no external package or checkpoint:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

It currently tests sequence validation, FASTA, regions, pooling, normalization,
cosine geometry, BLOSUM62, residue alignment, Sinkhorn OT, matrix agreement,
partial failures, exact limits, atomic artifacts, checksums, path traversal, and
SVG escaping.

Real-model conformance tests must be separately marked because they can download
large files and require specific devices.

Run the pinned ESM-2 8M conformance test explicitly:

```bash
HF_HOME=.model_cache PLMS_RUN_MODEL_TESTS=1 \
  python -m unittest tests.test_real_models -v
```

## Security notes

- Hugging Face loading sets `trust_remote_code=False`.
- Artifact and job names are restricted to safe characters.
- Artifact publication uses temporary files and atomic replacement.
- Access tokens are not part of request or provenance schemas.
- Model failures are returned structurally rather than hidden.
- Production deployments still need authentication, TLS, request-size limits,
  persistent job isolation, quotas, and retention controls.

## Development roadmap

The complete ordered backlog is maintained in [`ROADMAP.md`](ROADMAP.md). It is
the session-to-session source of truth for planned scientific, model, scaling,
API, security, and documentation work.

The next highest-value work is:

1. Add checkpoint-file checksums to the pinned ESM-2 8M provenance.
2. Resolve and store immutable model revisions for the other adapters.
3. Add content-addressed embedding caching and compressed binary artifacts.
4. Validate ESM-C, ProtT5, ProtBert, and Ankh one at a time.
5. Improve clustered heatmap ordering and add matrix CSV/NPZ exports.
6. Add persistent workers and job cancellation.
7. Implement explicit long-sequence window merging.
8. Add approximate neighbor search.
9. Introduce versioned, leakage-aware biological benchmark datasets.
10. Expand to MSA, structure-aware, generative, historical, and specialized PLMs.

Scientific validity and reproducibility take priority over a long list of model
names that have never passed conformance tests.
