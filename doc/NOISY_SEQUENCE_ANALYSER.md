# Noisy sequence analyser

`noisy_seq_analyser.py` compares reference proteins with matching records in a
directory of perturbed FASTA files. It uses the same sequence and embedding
scores as [`score_two_sequences.py`](../score_two_sequences.py). The
[`two-sequence example`](../examples/test_two_seqs.py) calls that scorer without
creating FASTA files.

Run the examples below from the repository root after installing the package.
The built-in `reference-deterministic-v1` model is a software fixture; select a
protein language model for biological analysis.

## Quick start

The included [reference FASTA](../examples/noisy_reference.fasta) has two
records. The [noisy FASTA directory](../examples/noisy_fastas) has two files;
three records match the reference headers.

```bash
python noisy_seq_analyser.py examples/noisy_reference.fasta examples/noisy_fastas \
  --models reference-deterministic-v1 --output noisy_sequence_scores.csv
```

The wide CSV contains one row per matched reference/noisy pair, in input order.
While the command runs, per-model CSVs are available in
`noisy_sequence_scores.models/`.

For one CSV per model, with rows flushed as pairs finish:

```bash
python noisy_seq_analyser.py examples/noisy_reference.fasta examples/noisy_fastas \
  --models reference-deterministic-v1 --model-wise \
  --output model_scores --workers 2 --batch-size 2
```

The result is `model_scores/reference-deterministic-v1.csv`. Rows may appear in
completion order; `pair_index` gives their zero-based input order.

## Input and output

- Records match on the **full FASTA header**. The analyser scans the noisy
  directory once, without recursion, and accepts `.fa`, `.faa`, `.fas`,
  `.fasta`, and `.fsa` files.
- By default, every reference header is considered. `--header protein_alpha`
  selects one exact header. A bare `--header` has the same effect as omitting it.
- A noisy file can omit reference headers. Duplicate matching headers in one
  noisy file, duplicate reference headers, and zero total matches are errors.
- Without `--model-wise`, the default output is
  `noisy_sequence_scores.csv`, plus live per-model CSVs in
  `noisy_sequence_scores.models/`. With `--model-wise`, the default output is
  the `noisy_sequence_scores/` directory.
- Every model CSV includes sequence scores, 14 embedding scores, a status, and
  an optional error. Long pairs are marked `skipped_long_sequence` by default.
  Results already flushed remain readable after a later failure. Rerunning a
  model overwrites its CSV; it does not resume a previous run.

## Flags

| Flag | Meaning | Default |
|---|---|---|
| `reference_fasta` | Reference FASTA path | Required |
| `noisy_fasta_dir` | Directory of noisy FASTAs | Required |
| `--header [HEADER]` | Select one full reference header | All headers |
| `--output PATH` | Wide CSV path or model-wise directory | `noisy_sequence_scores.csv` / `noisy_sequence_scores` |
| `--model-wise`, `--model_wise` | Write one CSV per model and clear its disposable checkpoint cache after each model | Off |
| `--models MODEL [MODEL ...]` | Framework model IDs | All ESM-1, ESM-2, and ESM-C IDs in the script |
| `--device {auto,cpu,cuda}` | Inference device | `auto` |
| `--precision {auto,float32,bfloat16,bf16}` | Inference precision | `auto` |
| `--workers N` | Concurrent complete-pair workers | CPU count minus one, capped at 8; minimum 1 |
| `--batch-size N` | Maximum sequences per embedding batch | `4` |
| `--max-batch-tokens N` | Maximum padded tokens per batch; longer sequences run alone | `2048` |
| `--torch-threads N` | PyTorch threads per operation | `1` |
| `--cache-dir DIR` | Disposable checkpoint cache for model-wise runs | Temporary directory |
| `--long-sequences {skip,error}` | Record over-limit pairs or stop | `skip` |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Log detail | `INFO` |

Use `python noisy_seq_analyser.py --help` to inspect the installed script's
options. `--models` without a value is invalid. Select a small model explicitly
for the first real-model run, for example `--models esm2-8m`. The default model
list includes checkpoints too large for a typical laptop.

## Execution and storage

Each worker embeds one reference/noisy pair, computes the scores, and releases
the embeddings. Equal sequences in a pair are embedded once. Active and queued
pairs are bounded to twice the worker count. `--batch-size` above two does not
combine unrelated pairs. CPU workers share weights but keep separate mutable
model state; GPU inference uses a lock to limit concurrent accelerator use.

No embedding arrays are written to disk. Wide mode stores scalar scores in a
temporary SQLite database while assembling the final CSV. Model-wise mode
directs checkpoint downloads to a disposable cache and empties **all contents**
of that directory after each model, even after a failure. Use a dedicated cache
directory only. With no `--cache-dir`, a temporary cache is created and removed.

Pairs above a model's native length are skipped unless `--long-sequences error`
is selected. Sequences are never silently truncated. If memory is tight, reduce
`--workers`, `--batch-size`, or `--max-batch-tokens`; increasing worker count
does not guarantee a proportional speedup.
