# Run workflows

Run commands from the repository root. Install the core package with
`python -m pip install -e .`; add extras as needed in the
[installation guide](../README.md#installation). Python 3.11 or newer is
required. The deterministic model shown below exercises the software without
downloading a checkpoint.

| Task | Entry point | Options and details |
|---|---|---|
| Inspect models | `plms-framework models` | Prints registered IDs and local availability. |
| Estimate resources | `plms-framework estimate examples/embed.json` | Pass a request JSON path or `-` for standard input. |
| Generate embeddings | `plms-framework embed examples/embed.json` | Pass a request JSON path or `-`. |
| Compare proteins | `plms-framework compare examples/compare.json` | Pass a request JSON path or `-`. |
| Compare two raw sequences | `python examples/test_two_seqs.py` | See [pair flags](#two-sequence-flags). |
| Score two sequences directly | `python examples/score_two_sequences.py MKTAYIAK MKTAYIAQ --models esm2-8m` | Same `--models`, `--device`, and `--precision` flags as the example; defaults to `esm2-8m`. |
| Analyse noisy FASTA files | `python examples/noisy_seq_analyser.py examples/noisy_reference.fasta examples/noisy_fastas --models reference-deterministic-v1` | See [analyser flags](NOISY_SEQUENCE_ANALYSER.md#flags). |
| Run the full Python demonstration | `python examples/python_ex.py` | See [demonstration flags](#python-demonstration-flags); requires the `all` runtime profile and checkpoints. |
| Start the API | `plms-framework serve` | See [server flags](#server-flags) and [API endpoints](../README.md#api-endpoints). |

The JSON files in [`examples/`](../examples) include CLI and API requests.
`embed.json`, `compare.json`, and `cross_model_*.json` can be passed to the CLI
commands above. `import_space.json`, `diagnostics.json`,
`transform_mean_center.json`, and `annotation_analysis.json` are API payloads
for their corresponding [endpoints](../README.md#api-endpoints); the CLI does
not expose those operations. Requests that refer to a source job need a
successful job ID from the running API. Replace their placeholder IDs before
submitting them. The import payload can be submitted directly.

For embedding and comparison fields and the Python equivalents, see the
[Python usage guide](PYTHON_USAGE.md#complete-request-option-reference).

## Two-sequence flags

[`examples/test_two_seqs.py`](../examples/test_two_seqs.py) prints five sequence
scores and, for each model, ten pooled-vector and four residue scores as JSON.
It calls [`score_two_sequences.py`](../examples/score_two_sequences.py), which shares the
scoring functions in [`noisy_seq_analyser.py`](../examples/noisy_seq_analyser.py).

```bash
python examples/test_two_seqs.py
python examples/test_two_seqs.py MKTAYIAK MKTAYIAQ \
  --models esm2-8m --device cpu --precision float32
```

| Argument | Meaning | Default |
|---|---|---|
| `sequence_a`, `sequence_b` | Raw protein sequences | `MKTAYIAK`, `MKTAYIAQ` |
| `--models MODEL [MODEL ...]` | Model IDs | `reference-deterministic-v1` |
| `--device {auto,cpu,cuda}` | Inference device | `auto` |
| `--precision {auto,float32,bfloat16,bf16}` | Inference precision | `auto` |

For a direct Python call, import `score_two_sequences` from
`examples.score_two_sequences`, then call it with
`score_two_sequences(a, b, models=("esm2-8m",), device="cpu")`. Its default
model is `esm2-8m`; the runnable example selects the deterministic fixture by
default so it works without model downloads.

## Python demonstration flags

`examples/python_ex.py` uses ESM-2 8M, ESM-C 300M, and ProGen2 small and writes results
to `python_ex_output/` by default. It covers representations, comparison,
cross-model methods, diagnostics, transformations, annotation analysis, and
visualizations. It requires installed model backends and checkpoint access.
Its synthetic inputs include a duplicate, a variant with nine substitutions,
and a shorter composition control; their labels are examples, not biological
annotations.

| Flag | Meaning | Default |
|---|---|---|
| `--device {auto,cpu,cuda}` | Common device | `auto` |
| `--precision {auto,float32,bfloat16,bf16}` | Common precision | `auto` |
| `--output-dir PATH` | Artifact directory | `python_ex_output` |

```bash
python examples/python_ex.py --device cpu --precision float32 --output-dir run_output
```

## Server flags

| Flag | Meaning | Default |
|---|---|---|
| `--host HOST` | Bind address | `127.0.0.1` |
| `--port PORT` | Bind port | `8000` |

```bash
plms-framework serve --host 127.0.0.1 --port 8000
```

The server needs the `api` extra. Its endpoints and request bodies are covered
in the [README](../README.md#api-endpoints).

## Automated checks

The [CI workflow](../.github/workflows/ci.yml) runs Ruff, the Python test suite, and
smoke checks for the CLI, pair example, and both noisy analyser output modes on
Python 3.11 and 3.12. It uses the deterministic adapter and downloads no model
checkpoints. The [Docs workflow](../.github/workflows/docs.yml) checks local
Markdown links, section references, and JSON syntax. The
[model check workflow](../.github/workflows/model-check.yml) is manual because it
downloads the ESM-2 8M checkpoint; it runs the conformance test and pair example.

Run the same core tests locally with:

```bash
python -m pip install -e '.[dev,api]'
python -m ruff check .
python -m pytest -q
python scripts/check_docs.py
```
