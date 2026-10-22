import atexit
import resource
import sys
from time import perf_counter

started_at = perf_counter()


def report_run_usage():
    elapsed = perf_counter() - started_at
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. Convert both to MiB.
    peak_ram_mib = peak_rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f"Peak process RAM: {peak_ram_mib:.2f} MiB ({peak_ram_mib / 1024:.2f} GiB)")


# Register before model imports so imports/loading count and failures also report usage.
atexit.register(report_run_usage)

from score_two_sequences import score_two_sequences

scores = score_two_sequences(
        "MKTAYIAK", "MKTAYIAQ",
        models=("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm1b-650m", "esm1v-650m-1", "esm1v-650m-2", "esm1v-650m-3", "esm1v-650m-4", "esm1v-650m-5", "esmc-300m", "esmc-600m"),
        device="auto",
)
print(scores)
print(scores["esm2-8m.mean.cosine_similarity"])
