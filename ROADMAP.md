# PLMs Framework — Master TODO

This is the ordered implementation backlog for the PLM comparison platform.

Checkbox meanings:

- `[ ]` not started
- `[~]` in progress
- `[x]` implemented and verified
- `[!]` blocked by an explicit external dependency or scientific decision

Priorities are `P0` (foundational), `P1` (high value), `P2` (important), and
`P3` (later expansion). An item is not complete until its implementation,
validation, documentation, examples, and tests agree.

## 0. Existing foundation

- [x] Capability-driven model registry
- [x] Deterministic reference adapter
- [x] Hugging Face encoder adapter
- [x] Dedicated ESM-C adapter surface
- [x] ESM-2 8M real-checkpoint conformance test
- [x] Mean, max, residue, region-mean, window-mean, and native representations
- [x] L2 normalization with original norms
- [x] Vector similarity and distance metrics
- [x] Sequence identity, k-mer, and BLOSUM controls
- [x] Residue alignment, best-match, and balanced Sinkhorn comparisons
- [x] Cross-model matrix correlation, CKA, neighbors, consensus, and disagreement
- [x] Single-sequence residue CKA and RSA
- [x] FastAPI model, embedding, comparison, benchmark, FASTA, job, and artifact routes
- [x] Atomic JSON/SVG/HTML artifacts with SHA-256 checksums
- [x] Exact all-to-all safety threshold
- [x] Partial model-failure isolation
- [x] Dependency-free core test path

## 1. External embedding spaces — P0

- [x] Add strict JSON import for precomputed vector and residue spaces
- [ ] Add NPZ upload with named arrays and ID manifest
- [ ] Add compressed NPZ export for framework-generated embeddings
- [ ] Add Parquet import/export for pooled vectors and metadata
- [ ] Add Zarr import/export for large residue tensors
- [ ] Add safe archive inspection before extraction
- [~] Add upload byte, record, dimension, and scalar-count limits (record, dimension, and scalar limits implemented; transport byte limit pending)
- [ ] Add streamed uploads rather than buffering complete files
- [ ] Add external-space schema versioning
- [x] Add source model/provider/revision/license/citation fields
- [x] Add declared pooling, layer, tokenization, and normalization metadata
- [ ] Add optional sequence SHA-256 and length declarations
- [ ] Verify sequence digests across imported spaces when available
- [ ] Add content checksum verification supplied by the client
- [ ] Add import dry-run/validation endpoint
- [ ] Add imported-space listing and inspection endpoints
- [ ] Add deletion/expiry policy for imported spaces
- [ ] Add persistent import catalog across API restarts
- [ ] Add conversion utility from EmmaEmb directory layout
- [ ] Add conversion utility from Hugging Face dataset columns
- [ ] Add conversion utility from common ESM `.pt` outputs
- [ ] Add conversion utility from per-protein `.npy` directories
- [x] Add clear trust boundary: imported values are validated structurally, not certified biologically

## 2. Embedding geometry diagnostics — P0

- [x] Vector-norm distribution and zero-vector detection
- [~] Duplicate and near-duplicate vector detection (exact duplicates implemented; near-duplicate search pending)
- [x] Per-dimension mean and variance summary
- [x] Average random-pair cosine diagnostic
- [x] Anisotropy score with dimensionality-aware interpretation
- [x] Participation ratio/effective dimensionality
- [ ] Singular-value spectrum and explained variance
- [x] Distance-concentration diagnostics
- [x] Hubness k-occurrence distribution
- [x] Robin Hood hubness index
- [ ] Mutual-neighbor rate
- [ ] Isolated-point detection
- [x] Diagnostic pair subsampling with fixed seed
- [ ] Confidence intervals for sampled diagnostics
- [x] Compare diagnostics before and after a transformation
- [x] Machine-readable diagnostic warnings
- [ ] Diagnostic visualizations and a summary panel

## 3. Explicit embedding transformations — P0

- [x] Mean-centering transformation
- [ ] Feature standardization transformation
- [ ] Whitening transformation with regularization
- [ ] Principal-component removal as an experimental method
- [ ] Unit-length normalization as a named pipeline stage
- [~] Fit transformations on declared reference/training partitions (explicit fit subsets supported; benchmark partition enforcement pending)
- [~] Prevent accidental data leakage into benchmark test partitions (all-sequence fit warning implemented; benchmark enforcement pending)
- [x] Reversible transformation records when mathematically possible
- [x] Transformation parameter artifacts
- [ ] Cache invalidation after transformations
- [x] Distinguish source normalization from framework-applied normalization
- [x] Require explicit opt-in; never silently alter imported vectors

## 4. Annotation-aware analysis — P0

- [~] Typed categorical and numerical annotation schema (both typed; categorical analysis implemented, numerical analysis pending)
- [x] Missing-value policy per annotation
- [x] Class-size and imbalance report
- [x] KNN label-consistency score per sample
- [x] Macro- and micro-averaged KNN consistency
- [x] Chance-adjusted KNN consistency
- [x] Per-class KNN performance
- [x] Class-mixing matrices
- [ ] Within-class versus between-class distance distributions
- [~] Retrieval precision@k, recall@k, MAP, and NDCG (precision and recall implemented; MAP and NDCG pending)
- [ ] One-versus-rest neighborhood enrichment
- [ ] Numerical-feature neighborhood smoothness
- [ ] Multi-label annotation support
- [ ] Hierarchical label support
- [ ] Ontology-aware partial credit
- [x] Minimum-class-size warnings
- [ ] Annotation-aware model disagreement reports
- [ ] Identify classes driving cross-model differences
- [ ] Metadata-stratified cross-model agreement

## 5. Sensitivity, uncertainty, and statistics — P0

- [ ] Protein-level bootstrap confidence intervals
- [ ] Stratified bootstrap by family/class
- [ ] Protein-label permutation tests
- [ ] Matrix-correlation permutation tests that respect pair dependence
- [ ] Neighbor-overlap confidence intervals
- [ ] CKA stability under sample resampling
- [ ] Model-ranking stability under sample resampling
- [ ] Sensitivity curves over k
- [ ] Sensitivity across distance metrics
- [ ] Sensitivity to class imbalance
- [ ] Sensitivity to controlled label noise
- [ ] Sensitivity to sequence subsampling
- [ ] Effect-size reporting
- [ ] Multiple-testing correction
- [ ] Seed and resampling-index provenance
- [ ] Small-sample power warnings
- [ ] Export full replicate distributions

## 6. Biological confounder controls — P0

- [ ] Partial matrix correlation controlling for sequence length
- [ ] Partial matrix correlation controlling for sequence identity
- [ ] Residualized agreement after multiple confounders
- [ ] Length-matched sampling
- [ ] Identity-binned reporting
- [ ] Remote-homology-only subsets
- [ ] Taxonomy-stratified reporting
- [ ] Family-stratified reporting
- [ ] Domain-count and architecture controls
- [ ] Disorder-content controls
- [ ] Membrane-protein stratification
- [ ] Structure-similarity controls when structures are supplied
- [ ] Report raw and controlled agreement side by side
- [ ] Detect confounder leakage in benchmark splits

## 7. Adapter SDK and conformance — P0

- [ ] Publish a stable adapter protocol
- [ ] Add adapter template generator
- [ ] Add `validate-adapter` CLI command
- [ ] Validate exact sequence-to-residue row counts
- [ ] Validate token-to-residue maps
- [ ] Validate padding invariance
- [ ] Validate batch invariance
- [ ] Validate repeated-run tolerance
- [ ] Validate selected layer behavior
- [ ] Validate CPU/GPU numerical agreement
- [ ] Validate precision behavior
- [ ] Validate maximum-length behavior
- [ ] Validate ambiguous-residue behavior
- [ ] Validate empty and malformed inputs
- [ ] Validate unload/reload behavior
- [ ] Validate upstream revision resolution
- [ ] Record checkpoint file checksums
- [ ] Add signed conformance result manifests
- [ ] Add conformance tiers: structural, numerical, biological, operational
- [ ] Refuse “tested” status without retained evidence
- [ ] Add third-party adapter/plugin discovery
- [ ] Add adapter API compatibility version

## 8. Model coverage — P1

- [ ] Validate ESM-2 35M
- [ ] Validate ESM-2 150M
- [ ] Validate ESM-2 650M
- [ ] Validate ESM-2 3B on suitable hardware
- [ ] Validate ESM-2 15B on suitable hardware
- [ ] Validate ESM-C 300M
- [ ] Validate ESM-C 600M
- [ ] Validate ESM-C 6B on suitable hardware
- [ ] Validate ProtT5 XL UniRef50
- [ ] Validate ProtBert BFD
- [ ] Validate Ankh Base
- [ ] Validate Ankh Large
- [ ] Add ProstT5
- [ ] Add ESM-1b
- [ ] Add ESM-1v ensemble variants
- [ ] Add ESM MSA Transformer capability class
- [ ] Add ESM-IF1 inverse-folding capability class
- [ ] Add ESM-3 multimodal capability class
- [ ] Add SaProt structure-token adapter
- [ ] Add ProSST structure-aware adapter
- [ ] Add antibody-specific model category
- [ ] Add nucleotide/protein joint model category only if scope is explicitly expanded
- [ ] Track deprecated and superseded checkpoints
- [ ] Generate registry documentation from manifests

## 9. Long-sequence execution — P1

- [ ] Model-aware native length resolution
- [ ] Explicit overlapping-window policy
- [ ] Configurable window size and overlap
- [ ] Deterministic left-to-right window plan
- [ ] Optional bidirectional plan for evaluation
- [ ] Edge-weighted overlap reconstruction
- [ ] Residue coverage counts
- [ ] Reject uncovered residues
- [ ] Pool only after residue reconstruction
- [ ] Record native versus approximated context
- [ ] Compare windowed and native outputs on sequences fitting both paths
- [ ] Boundary-artifact diagnostics
- [ ] Global-context-loss warning
- [ ] Model-specific global-token handling
- [ ] Window batching without changing output order
- [ ] Long-sequence resource estimates
- [ ] Long-sequence conformance fixtures

## 10. Caching and storage — P1

- [ ] Content-addressed embedding cache
- [ ] Cache key includes sequence digest and immutable model revision
- [ ] Cache key includes layer, precision, adapter, and preprocessing
- [ ] Cache key includes long-sequence policy
- [ ] Separate raw-residue cache from derived pooling cache
- [ ] Cache integrity checksum validation
- [ ] Atomic cache publication
- [ ] Concurrent cache-write locking
- [ ] Least-recently-used eviction
- [ ] Configurable storage quota
- [ ] Cache inspection CLI/API
- [ ] Cache hit/miss provenance
- [ ] Compressed chunked residue storage
- [ ] Memory-mapped pooled matrices
- [ ] Sparse neighbor artifact format
- [ ] Matrix CSV and compressed binary exports
- [ ] Artifact retention and expiry
- [ ] Garbage collection for orphan artifacts

## 11. Large-scale similarity — P1

- [ ] Blockwise exact matrix computation
- [ ] Stream matrix blocks to disk
- [ ] Avoid JSON for large dense matrices
- [ ] HNSW approximate-neighbor backend
- [ ] FAISS CPU backend
- [ ] Optional FAISS GPU backend
- [ ] Explicit recall-versus-speed parameters
- [ ] ANN validation against exact neighbors on a sample
- [ ] Store ANN index provenance
- [ ] Top-k-only comparison mode
- [ ] Radius-neighbor mode
- [ ] Incremental index updates
- [ ] Sharded embedding collections
- [ ] Distributed matrix computation interface
- [ ] Resource planner chooses only user-authorized modes
- [ ] Never silently replace exact execution with approximate execution

## 12. Cross-model methodology — P1

- [ ] Layer-by-layer CKA
- [ ] Layer correspondence heatmap across model families
- [ ] Kernel CKA with explicit kernel parameters
- [ ] Unbiased HSIC/CKA estimator option
- [ ] Partial relationship-matrix correlation
- [ ] Weighted relationship agreement
- [ ] Annotation-stratified agreement
- [ ] Per-sequence neighborhood-disagreement score
- [ ] Mutual-neighbor agreement
- [ ] Model clustering from agreement matrices
- [ ] Consensus reliability per sequence pair
- [ ] Robust consensus using median ranks
- [ ] Leave-one-model-out consensus sensitivity
- [ ] Detect redundant models
- [ ] Identify complementary model ensembles
- [ ] Compare model ensembles without mixing raw coordinate spaces
- [ ] Residue agreement restricted to named regions
- [ ] Residue agreement by secondary-structure segment
- [ ] Cross-model contact-map agreement when available
- [ ] Cross-model attention analysis only with strong caveats

## 13. Residue and transport methods — P1

- [ ] Affine-gap embedding alignment
- [ ] Configurable, recorded alignment penalties
- [ ] Gap-aware unbalanced optimal transport
- [ ] Banded residue alignment
- [ ] Local alignment top-N non-overlapping hits
- [ ] Residue correspondence confidence
- [ ] Region-to-region comparison
- [ ] Motif query against sequence collections
- [ ] Residue saliency import contract
- [ ] Domain-boundary-aware aggregation
- [ ] Compare residue methods against known alignments
- [ ] Memory limits for L1×L2 matrices
- [ ] Blockwise residue similarity
- [ ] Export alignment paths in tabular formats

## 14. Biological benchmark engine — P1

- [ ] Versioned dataset manifest format
- [ ] Dataset download policy and checksum verification
- [ ] User-supplied local benchmark support
- [ ] Family-aware train/test splitting
- [ ] Sequence-identity-aware splitting
- [ ] Temporal splitting
- [ ] Pretraining-overlap declaration
- [ ] Frozen-representation track
- [ ] Native zero-shot track
- [ ] Native best-practice track
- [ ] Learned-pooling track reported separately
- [ ] Efficiency/cost track
- [ ] Remote-homology benchmark
- [ ] Protein-family retrieval benchmark
- [ ] Function benchmark
- [ ] Localization benchmark
- [ ] Secondary-structure benchmark
- [ ] Disorder benchmark
- [ ] Domain-boundary benchmark
- [ ] Protein-interaction benchmark
- [ ] Variant-effect benchmark
- [ ] Structure-similarity benchmark
- [ ] Bootstrap benchmark uncertainty
- [ ] Benchmark result schema and leaderboard export

## 15. Reports and visualization — P1

- [ ] Portable self-contained HTML report
- [ ] Dataset quality overview
- [ ] Model manifest cards
- [ ] Diagnostic dashboard
- [ ] Annotation-aware plots
- [ ] Interactive matrix inspection
- [ ] Hoverable disagreement pairs
- [ ] Linked sequence/region views
- [ ] Layer-comparison heatmaps
- [ ] Confidence intervals on plots
- [ ] Accessible color palettes
- [ ] Colorblind-safe defaults
- [ ] Large-label suppression and search
- [ ] Static PNG/PDF export
- [ ] Publication-size presets
- [ ] Download buttons for underlying data
- [ ] Report schema version and complete provenance
- [ ] Clearly mark exploratory nonlinear projections

## 16. API and client usability — P1

- [ ] Version all public response schemas
- [ ] Generate a typed Python client
- [ ] Generate an R client or minimal R examples
- [ ] Paginate model/job/space listings
- [ ] Idempotency keys for submissions
- [ ] Request dry-run validation
- [ ] Server-side request-size enforcement
- [ ] Streaming upload/download
- [ ] Server-sent job progress events
- [ ] Job cancellation endpoint
- [ ] Job retry endpoint
- [ ] Structured error catalog
- [ ] Batch request manifests
- [ ] Webhook notification with explicit authorization
- [ ] CLI parity for all API operations
- [ ] OpenAPI examples for every major workflow
- [ ] Backward-compatibility tests

## 17. Persistent execution and operations — P1

- [ ] Persistent job database
- [ ] Persistent worker queue
- [ ] Restart-safe job recovery
- [ ] GPU-aware scheduler
- [ ] Per-device concurrency limits
- [ ] Model residency/unload policy
- [ ] Admission control using resource estimates
- [ ] CPU fallback policy
- [ ] Job timeouts
- [ ] Artifact lifecycle states
- [ ] Structured logs without raw private sequences
- [ ] Metrics and tracing
- [ ] Health checks for workers and storage
- [ ] Deployment reference for one GPU
- [ ] Container image with pinned dependencies
- [ ] Kubernetes deployment example only after single-node maturity

## 18. Security and privacy — P1

- [ ] Authentication
- [ ] Authorization per job and artifact
- [ ] Tenant isolation
- [ ] Request and storage quotas
- [ ] TLS deployment guidance
- [ ] Private-sequence retention controls
- [ ] Configurable sequence redaction from results
- [ ] Secret scanning and dependency audit
- [ ] Uploaded archive/path traversal protection
- [ ] Model remote-code policy enforcement
- [ ] Gated-model credential handling outside request bodies
- [ ] Audit trail for downloads and deletions
- [ ] Threat model document
- [ ] Security response policy

## 19. Reproducibility and provenance — P1

- [ ] Resolve all upstream aliases to immutable commits
- [ ] Checksum every checkpoint file
- [ ] Record package lock/environment fingerprint
- [ ] Record hardware and accelerator details
- [ ] Record deterministic-algorithm settings
- [ ] Record every random seed
- [ ] Record input dataset manifest checksum
- [ ] Record transformation parameters
- [ ] Record cache source and integrity result
- [ ] Replay command/API payload artifact
- [ ] Verify a completed run against its manifest
- [ ] Export RO-Crate or another research-object format
- [ ] Stable citation bundle per model and dataset

## 20. Testing and quality — P0/P1

- [ ] Property-based tests for metric invariants
- [ ] Golden numerical fixtures for every metric
- [ ] Cross-backend parity fixtures
- [ ] API schema snapshot tests
- [ ] Artifact schema tests
- [ ] Large-input resource-limit tests
- [ ] Concurrency and atomicity tests
- [ ] Cache corruption tests
- [ ] Fuzz FASTA and import parsers
- [ ] Security regression tests
- [ ] Performance regression benchmarks
- [ ] Memory regression benchmarks
- [ ] Test supported Python versions in CI
- [ ] Test CPU installation in CI
- [ ] Optional GPU CI tier
- [ ] Documentation code-block tests
- [ ] Example request smoke tests
- [ ] Mutation testing for scientific validation paths

## 21. Documentation and community — P2

- [ ] Architecture decision records
- [ ] Scientific-method reference pages
- [ ] Adapter author guide
- [ ] Deployment guide
- [ ] Privacy guide
- [ ] Benchmark author guide
- [ ] Troubleshooting matrix
- [ ] Versioned changelog
- [ ] Contribution guide
- [ ] Code of conduct
- [ ] Issue templates
- [ ] Model-adapter request template
- [ ] Reproducible tutorial notebooks
- [ ] Small real biological example dataset
- [ ] EmmaEmb interoperability tutorial
- [ ] Citation file and model-specific citation export

## 22. Later research directions — P2/P3

- [ ] Learned pooling with leakage-safe training
- [ ] Multi-layer pooling
- [ ] Supervised metric learning as a separate track
- [ ] Calibration of similarity to biological relatedness
- [ ] Ensemble selection under compute budgets
- [ ] Active sampling of informative proteins
- [ ] MSA-depth sensitivity analysis
- [ ] Structure-token and sequence-token joint comparisons
- [ ] Multimodal sequence/structure consistency
- [ ] Contact and attention-derived representations
- [ ] Variant embedding trajectories
- [ ] Taxonomic bias analysis
- [ ] Low-complexity and disorder bias analysis
- [ ] Pretraining contamination studies
- [ ] Energy/carbon reporting when measurements are available

## Immediate execution order

1. Finish strict JSON external-space import and connect it to cross-comparison.
2. Add external-space examples and complete validation tests.
3. Implement norm, anisotropy, participation-ratio, and hubness diagnostics.
4. Add explicit mean-centering with before/after diagnostics.
5. Add typed annotations and KNN label-consistency analysis.
6. Add sensitivity over k and class-imbalance-aware summaries.
7. Add bootstrap confidence intervals and protein-level permutation tests.
8. Add content-addressed embedding caching.
9. Add adapter conformance CLI and validate real adapters one by one.
10. Add safe long-sequence overlap reconstruction.
