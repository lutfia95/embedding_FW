# Noisy sequence analyser

Run from `embedding_FW` in your framework environment:

```bash
PYTHONPATH=src python noisy_seq_analyser.py reference.fasta noisy_fastas/ \
  --model_wise --output model_scores/ \
  --models esm2-8m \
  --workers 8 --torch-threads 1 --batch-size 2
```

Use your actual reference FASTA and directory of perturbation FASTAs. Omit
`--models` to run the default ESM models. Matching uses the full FASTA header.
Both `--model_wise` and `--model-wise` are accepted.

Each of the `--workers` threads receives a complete reference/noisy pair. It
embeds up to two sequences, calculates the existing metrics, and releases the
embeddings. Identical sequences within a pair are embedded only once. A
reference is recomputed for each pair so there is no growing embedding cache.
The active and queued work is bounded to twice the worker count.

CPU workers share model weights while keeping separate tokenizers and mutable
model caches. `--torch-threads` controls PyTorch threads per operation and
defaults to one to avoid multiplying thread pools across workers. GPU model
inference uses a shared lock to bound accelerator memory; pair scoring overlaps
in the worker threads. Python metric loops still have the interpreter's GIL;
thread count alone does not guarantee proportional speedup.

`--batch-size` and `--max-batch-tokens` bound batches within a pair. A batch size
above two does not combine unrelated pairs. Reduce `--workers` if the active
pairs consume too much memory. No embedding arrays are written to disk.

With `--model_wise`, results are appended and flushed immediately to
`model_scores/<model-id>.csv`. Rows follow completion order. `pair_index` is the
zero-based input pair index; the existing metric column names are preserved.
Every result includes `<model-id>.status` and an optional `<model-id>.error`.
These CSVs remain readable during execution and survive a subsequent failure.
Existing files for a model are overwritten when that model starts; rerunning
does not resume a previous run.

Without `--model_wise`, `--output scores.csv` produces the final wide report in
input order. Live results are available in `scores.models/<model-id>.csv`.
Only scalar scores are stored in a temporary SQLite database for the final
merge; that database is removed afterward. The command does not keep all score
rows or embeddings in memory. Python callers that omit `output_path` retain the
legacy in-memory list return value; streaming calls return the input pair count.

Pairs exceeding a model's native length are recorded as
`skipped_long_sequence`, with blank model scores, and processing continues.
There is no automatic truncation. Use `--long-sequences error` to stop instead.
Other inference/scoring errors stop the run, preserving scores already flushed.

In model-wise mode, optional `--cache-dir model_cache/` controls **checkpoint
downloads**, separately from pair embeddings. Its existing behavior is retained:
the dedicated cache directory is emptied after each model, including on failure.
Without that option, a temporary checkpoint directory is used and cleaned.
