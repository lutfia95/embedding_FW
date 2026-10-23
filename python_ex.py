#!/usr/bin/env python3
"""Complete ESM-2, ESM-C, and ProGen2 comparison example with explained output.

Run this example from the repository root after the complete installation:

    python3.12 -m pip install -r requirements.txt
    python3.12 python_ex.py

The first run downloads ``facebook/esm2_t6_8M_UR50D``, ``biohub/ESMC-300M``,
and ``hugohrban/progen2-small``. ESM-C currently requires Python 3.12+.
The example uses strict failure handling: all three models must finish before
cross-model results are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plms_framework import (
    AnnotationAnalysisSpec,
    DiagnosticSpec,
    Pipeline,
    TransformationSpec,
)
from plms_framework.annotations import analyze_categorical_annotations
from plms_framework.artifacts import ArtifactStore
from plms_framework.diagnostics import diagnose_embeddings
from plms_framework.errors import FrameworkError
from plms_framework.models import (
    MissingAnnotationPolicy,
    RepresentationMethod,
    TransformationMethod,
)
from plms_framework.serialization import request_from_dict
from plms_framework.transformations import transform_embeddings
from plms_framework.visualization import render_requested


MODEL_IDS = ("esm2-8m", "esmc-300m", "progen2-small")
JOB_ID = "esm2-esmc-complete-example"

# S1 and S2 are exact duplicates. S3 has three substitutions relative to S1.
# S4 has the same length but deliberately unrelated, low-complexity composition.
SEQUENCES = (
    {
        "id": "reference",
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANNMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANNMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANN",
        "metadata": {"role": "reference", "expected_group": "reference_like"},
        "regions": [
            {"name": "n_terminal", "start": 1, "end": 12},
            {"name": "middle", "start": 13, "end": 24},
            {"name": "c_terminal", "start": 25, "end": 36},
        ],
    },
    {
        "id": "exact_duplicate",
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANNMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANNMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANN",
        "metadata": {"role": "exact duplicate", "expected_group": "reference_like"},
        "regions": [
            {"name": "n_terminal", "start": 1, "end": 12},
            {"name": "middle", "start": 13, "end": 24},
            {"name": "c_terminal", "start": 25, "end": 36},
        ],
    },
    {
        "id": "similar_variant",
        "sequence": "MKTAYIAKQRQVSFVKSHFTRQLEERLGLVEVQANNMKTAYIAKQRQVSFVKSHFTRQLEERLGLVEVQANNMKTAYIAKQRQVSFVKSHFTRQLEERLGLVEVQANN",
        "metadata": {"role": "three substitutions", "expected_group": "reference_like"},
        "regions": [
            {"name": "n_terminal", "start": 1, "end": 12},
            {"name": "middle", "start": 13, "end": 24},
            {"name": "c_terminal", "start": 25, "end": 36},
        ],
    },
    {
        "id": "different_control",
        "sequence": "GPGSGPGSGPGSGPGSGMKTAYIAKQRQVSPGSGPGSGPGSGPGSGPGSTAYIAKQRQVSAAANMMM",
        "metadata": {"role": "unrelated control", "expected_group": "composition_control"},
        "regions": [
            {"name": "n_terminal", "start": 1, "end": 12},
            {"name": "middle", "start": 13, "end": 24},
            {"name": "c_terminal", "start": 25, "end": 36},
        ],
    },
)

REPRESENTATIONS = (
    {"method": "mean", "normalize": "l2"},
    {"method": "max", "normalize": "l2"},
    {"method": "residue", "normalize": "l2"},
    {"method": "region_mean", "normalize": "l2"},
    {
        "method": "window_mean",
        "normalize": "l2",
        "window_size": 12,
        "window_step": 6,
    },
)

# This is every currently implemented metric. The pipeline applies pooled-vector
# metrics to both mean and max representations, sequence metrics once, and
# residue metrics to each model's residue representation.
METRICS = (
    "cosine_similarity",
    "cosine_distance",
    "angular_distance",
    "correlation_distance",
    "euclidean_distance",
    "length_ratio",
    "sequence_identity",
    "kmer_jaccard",
    "blosum62_global",
    "blosum62_local",
    "residue_global_alignment",
    "residue_local_alignment",
    "residue_best_match",
    "residue_optimal_transport",
)

CROSS_MODEL_METHODS = (
    "matrix_spearman",
    "matrix_pearson",
    "linear_cka",
    "neighbor_overlap",
    "consensus_rank",
    "pair_disagreement",
    "residue_cka",
    "residue_rsa",
)

VISUALIZATIONS = (
    "clustered_heatmap",
    "dendrogram",
    "pcoa",
    "umap",
    "nearest_neighbor_graph",
    "residue_similarity_map",
    "model_agreement",
    "distance_distribution",
    "model_matrix_small_multiples",
    "consensus_heatmap",
    "model_disagreement",
    "residue_model_agreement",
)

METRIC_EXPLANATIONS = {
    "cosine_similarity": "higher means a smaller angle between pooled vectors (-1 to 1)",
    "cosine_distance": "lower is closer; defined as 1 - cosine similarity",
    "angular_distance": "lower is closer; angular separation scaled to 0..1",
    "correlation_distance": "lower is closer after centering each vector's features",
    "euclidean_distance": "lower is closer; L2 distance between normalized pooled vectors",
    "length_ratio": "higher is closer; shorter length divided by longer length",
    "sequence_identity": "higher is closer; global alignment identity per alignment column",
    "kmer_jaccard": "higher is closer; overlap between sets of unique 3-mers",
    "blosum62_global": "higher is closer; normalized global BLOSUM62 alignment score",
    "blosum62_local": "higher is closer; normalized local BLOSUM62 alignment score",
}


def build_request(*, device: str = "auto", precision: str = "auto"):
    """Build the complete three-model comparison request used by this example."""
    return request_from_dict(
        {
            "models": list(MODEL_IDS),
            "sequences": list(SEQUENCES),
            "representations": list(REPRESENTATIONS),
            "metrics": list(METRICS),
            "visualizations": list(VISUALIZATIONS),
            "execution": {
                "device": device,
                "precision": precision,
                "all_to_all": "exact",
                "on_long_sequence": "error",
                "allow_partial": False,
                "max_exact_sequences": 100,
            },
            "cross_model": {
                "enabled": True,
                "methods": list(CROSS_MODEL_METHODS),
                "representation": "mean",
                "neighbor_k": [1, 2, 3],
                "max_disagreements": 6,
            },
        },
        compare=True,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def short_labels(sequence_order: list[str]) -> dict[str, str]:
    return {sequence_id: f"S{index}" for index, sequence_id in enumerate(sequence_order, 1)}


def print_matrix(
    title: str,
    matrix: list[list[Any]],
    row_names: list[str],
    *,
    explanation: str,
) -> None:
    """Print a compact numeric matrix while retaining the authoritative row order."""
    print(f"\n{title}")
    print(f"  Meaning: {explanation}.")
    width = max(10, *(len(name) + 1 for name in row_names))
    print(" " * width + "".join(f"{name:>11}" for name in row_names))
    for name, row in zip(row_names, matrix, strict=True):
        values = "".join(f"{number(value):>11}" for value in row)
        print(f"{name:<{width}}{values}")


def print_sequence_design(sequence_order: list[str]) -> None:
    labels = short_labels(sequence_order)
    section("1. Experimental design")
    print("The four inputs are a deliberate sanity check for every score:")
    for item in SEQUENCES:
        print(
            f"  {labels[item['id']]} = {item['id']:<18} length={len(item['sequence'])}  "
            f"role={item['metadata']['role']}"
        )
        print(f"       {item['sequence']}")
    print("\nExpected pattern:")
    print("  - S1 and S2 should be identical in deterministic evaluation mode.")
    print("  - S3 differs from S1/S2 at exactly three positions and should remain close.")
    print("  - S4 has unrelated, low-complexity composition and should usually be far away.")
    print("  - All sequences have equal length, so length_ratio alone cannot distinguish them.")
    print("  - Regions are residues 1-12, 13-24, and 25-36; windows use size 12, step 6.")


def print_estimate(estimate: dict) -> None:
    section("2. Pre-run resource estimate")
    print(
        f"Sequences: {estimate['sequence_count']}; unique unordered pairs: "
        f"{estimate['pair_count']}; bytes per 4x4 float64 matrix: "
        f"{estimate['matrix_bytes_float64_per_metric']}"
    )
    for model in estimate["models"]:
        print(
            f"  {model['model']}: {model['residues']} total residues, "
            f"residue-output estimate={human_bytes(model['embedding_bytes_float32'])}, "
            f"parameters={model['parameters']:,}, "
            f"checkpoint estimate={human_bytes(model['checkpoint_bytes_estimate'])}"
        )
    print(
        "These are planning estimates, not measured peak RAM/VRAM. Temporary activations, "
        "backend buffers, dtype, and checkpoint packaging can change actual usage."
    )


def check_backends(pipeline: Pipeline) -> bool:
    section("3. Backend availability")
    ready = True
    for model_id in MODEL_IDS:
        adapter = pipeline.registry.adapter(model_id)
        availability = adapter.availability()
        manifest = adapter.manifest
        state = "ready" if availability.available else "NOT READY"
        print(
            f"  {model_id}: {state}; backend={manifest.backend}; artifact={manifest.artifact}; "
            f"dimension={manifest.embedding_dimension}"
        )
        if not availability.available:
            ready = False
            print(f"    reason_code={availability.reason_code}")
            print(f"    detail={availability.detail}")
    if not ready:
        print("\nThe complete example needs all three models. Install with:")
        print("  python3.12 -m pip install -r requirements.txt")
        print("Then rerun this file under Python 3.12+. No one-model fallback is used because")
        print("that would make the requested cross-model results unavailable.")
    return ready


def print_model_outputs(result: dict) -> None:
    section("4. Successful models, provenance, and representations")
    print(
        "Each adapter ran once at model_default. Boundary/padding tokens were removed before "
        "the framework derived all five requested representation types."
    )
    for model_id, output in result["models"].items():
        manifest = output["manifest"]
        provenance = output["provenance"]
        print(f"\n{model_id} — {manifest['display_name']}")
        print(
            f"  backend={manifest['backend']}; artifact={manifest['artifact']}; "
            "revision="
            f"{provenance.get('resolved_revision') or provenance.get('requested_revision')}"
        )
        print(
            f"  actual device={provenance.get('device', 'not reported')}; "
            f"actual dtype={provenance.get('dtype', 'not reported')}; "
            f"conformance={manifest['conformance']}"
        )
        by_method: dict[str, list[dict]] = {}
        for item in output["representations"]:
            by_method.setdefault(item["method"], []).append(item)
        for method, items in by_method.items():
            shapes = sorted({str(tuple(item["shape"])) for item in items})
            print(
                f"  {method:<12} records={len(items)}, shapes={', '.join(shapes)}, "
                f"normalization={items[0]['normalization']}"
            )
        print("  mean/max: one pooled vector per protein.")
        print("  residue: one vector per amino acid; each row is L2-normalized here.")
        print("  region_mean/window_mean: local pooled vectors with recorded coordinates.")
    print(
        "\nThe optional native representation is not requested: neither selected model "
        "advertises a backend-native pooled vector. Raw coordinates are never compared "
        "directly between ESM-2 and ESM-C because their dimensions and learned spaces differ."
    )


def print_sequence_baselines(result: dict) -> None:
    section("5. Sequence-only baselines")
    order = result["sequence_order"]
    labels = short_labels(order)
    compact = [labels[item] for item in order]
    print("These matrices use amino-acid strings only; neither neural model affects them.")
    for metric, matrix in result["sequence_baselines"].items():
        print_matrix(
            metric,
            matrix,
            compact,
            explanation=METRIC_EXPLANATIONS[metric],
        )
    print(
        "\nSanity check: S1/S2 should score 1 for all similarities. S1/S3 should be "
        "strongly similar. The equal length ratio of S4 demonstrates why length is only a "
        "baseline and not evidence of sequence or functional relatedness."
    )


def print_pooled_matrices(result: dict) -> None:
    section("6. Within-model pooled embedding comparisons")
    order = result["sequence_order"]
    labels = short_labels(order)
    compact = [labels[item] for item in order]
    print(
        "Every matrix is within one model and one representation space. Mean and max pooling "
        "are shown separately; do not compare raw vector coordinates across the two PLMs."
    )
    for model_id, output in result["models"].items():
        print(f"\n--- {model_id} ---")
        for matrix_key, matrix in output["matrices"].items():
            metric = matrix_key.rsplit(":", 1)[1]
            print_matrix(
                matrix_key,
                matrix,
                compact,
                explanation=METRIC_EXPLANATIONS[metric],
            )


def print_residue_results(result: dict) -> None:
    section("7. Residue-level comparisons")
    print(
        "For every protein pair, each model builds an L1 x L2 residue cosine map. Global and "
        "local alignment optimize paths through it; best-match is bidirectional nearest-residue "
        "similarity; optimal transport is balanced entropic Sinkhorn similarity. Higher values "
        "mean stronger residue-level agreement for all four columns below."
    )
    for model_id, output in result["models"].items():
        print(f"\n{model_id}")
        print(
            f"  {'pair':<39}{'shape':>10}{'global':>11}{'local':>11}"
            f"{'best':>11}{'transport':>12}"
        )
        for pair, values in output["residue_pairs"].items():
            global_score = values["residue_global_alignment"]["normalized_score"]
            local_score = values["residue_local_alignment"]["normalized_score"]
            best_match = values["residue_best_match"]
            transport = values["residue_optimal_transport"]["similarity"]
            shape = "x".join(str(item) for item in values["similarity_shape"])
            print(
                f"  {pair:<39}{shape:>10}{global_score:>11.5f}{local_score:>11.5f}"
                f"{best_match:>11.5f}{transport:>12.5f}"
            )
    print(
        "Full alignment paths, matched-residue means, transport costs, and similarity maps are "
        "retained in comparison.json; printing every path here would obscure the summary."
    )


def print_cross_model_results(result: dict) -> None:
    section("8. Cross-model agreement")
    cross = result["cross_model"]
    print(cross["scientific_guardrail"] + ".")
    print(
        "The comparison asks whether models preserve similar relationships among proteins. "
        "It does not take cosine similarity between an ESM-2 vector and an ESM-C vector."
    )
    comparisons = cross["comparisons"]
    explanations = {
        "matrix_spearman": "rank correlation between the models' six off-diagonal pair scores",
        "matrix_pearson": "linear correlation between the models' six pair scores",
        "linear_cka": "centered-kernel alignment; valid even when embedding dimensions differ",
    }
    for method in ("matrix_spearman", "matrix_pearson", "linear_cka"):
        item = comparisons[method]
        if item.get("available"):
            print_matrix(
                method,
                item["values"],
                item["model_order"],
                explanation=explanations[method],
            )
        else:
            print(f"\n{method}: unavailable — {item.get('reason')}")

    overlap = comparisons["neighbor_overlap"]
    if overlap.get("available"):
        for k, matrix in overlap["by_k"].items():
            print_matrix(
                f"neighbor_overlap@k={k}",
                matrix,
                overlap["model_order"],
                explanation="mean Jaccard overlap of each protein's within-model neighbor sets",
            )

    consensus = comparisons["consensus_rank"]
    if consensus.get("available"):
        labels = short_labels(consensus["sequence_order"])
        print_matrix(
            "consensus_rank",
            consensus["values"],
            [labels[item] for item in consensus["sequence_order"]],
            explanation=consensus["scale"],
        )

    disagreement = comparisons["pair_disagreement"]
    if disagreement.get("available"):
        print("\nPairs on which model rankings disagree most (0 means identical rank percentile):")
        for item in disagreement["top_pairs"]:
            ranks = ", ".join(
                f"{model}={rank:.3f}"
                for model, rank in item["model_rank_percentiles"].items()
            )
            print(
                f"  {item['left']} :: {item['right']}: range={item['rank_range']:.3f}; {ranks}"
            )

    for method in ("residue_cka", "residue_rsa"):
        item = comparisons[method]
        meaning = (
            "cross-model CKA over residue observations"
            if method == "residue_cka"
            else "Spearman agreement between within-protein residue-similarity structures"
        )
        print(f"\n{method}: {meaning}.")
        for sequence_id, agreement in item["by_sequence"].items():
            if agreement.get("available"):
                off_diagonal = agreement["values"][0][1]
                print(
                    f"  {sequence_id:<18} residues={agreement['residue_count']:<3} "
                    f"ESM-2 vs ESM-C={number(off_diagonal)}"
                )
            else:
                print(f"  {sequence_id}: unavailable — {agreement['reason']}")
    for warning in cross.get("warnings", []):
        print(f"  warning: {warning}")


def run_and_print_diagnostics(result: dict) -> dict:
    section("9. Embedding geometry diagnostics")
    diagnostics = diagnose_embeddings(
        result["models"],
        result["sequence_order"],
        DiagnosticSpec(
            representation=RepresentationMethod.MEAN,
            layer="model_default",
            sample_pairs=100,
            hubness_k=2,
            seed=42,
            max_exact_hubness_sequences=100,
        ),
    )
    print(
        "Diagnostics describe geometry, not biological quality. With only four deliberately "
        "chosen proteins, treat anisotropy, effective dimension, and hubness as demonstrations."
    )
    for model_id, item in diagnostics["models"].items():
        if not item.get("available"):
            print(f"  {model_id}: unavailable — {item['reason']}")
            continue
        print(f"\n{model_id}")
        print(
            f"  dimension={item['embedding_dimension']}; "
            f"duplicate vectors={item['duplicates']['duplicate_vector_count']}; "
            f"participation ratio={number(item['dimensions']['participation_ratio'])}"
        )
        print(
            f"  average pairwise cosine={number(item['anisotropy']['average_pairwise_cosine'])}; "
            f"anisotropy score={number(item['anisotropy']['anisotropy_score'])}; "
            "mean-centered cosine="
            f"{number(item['anisotropy']['mean_centered_average_pairwise_cosine'])}"
        )
        hubness = item["hubness"]
        print(
            f"  hubness k={hubness.get('k')}; "
            f"Robin-Hood index={number(hubness.get('robin_hood_index'))}; "
            f"occurrence range={hubness.get('occurrence_min')}..{hubness.get('occurrence_max')}"
        )
        for warning in item["warnings"]:
            print(f"  warning: {warning}")
    print(f"\nGuardrail: {diagnostics['scientific_guardrail']}.")
    return diagnostics


def run_and_print_transformation(result: dict) -> dict:
    section("10. Immutable mean-centering transformation")
    centered = transform_embeddings(
        result["models"],
        result["sequence_order"],
        TransformationSpec(
            method=TransformationMethod.MEAN_CENTER,
            representation=RepresentationMethod.MEAN,
            layer="model_default",
            fit_sequence_ids=("reference", "similar_variant", "different_control"),
            sample_pairs=100,
            hubness_k=2,
            seed=42,
        ),
    )
    print(
        "A mean fitted on S1, S3, and S4 is subtracted without modifying the source result. "
        "S2 is deliberately held out to demonstrate an explicit fit subset."
    )
    for source, derived in centered["transformation"]["source_model_map"].items():
        parameters = centered["transformation_parameters"][derived]
        before = centered["diagnostics"]["before"]["models"][source]
        after = centered["diagnostics"]["after"]["models"][derived]
        print(f"\n  {source} -> {derived}")
        print(f"    parameter SHA-256={parameters['sha256']}")
        print(
            "    average cosine before/after="
            f"{number(before['anisotropy']['average_pairwise_cosine'])} / "
            f"{number(after['anisotropy']['average_pairwise_cosine'])}"
        )
    print(
        "The checksum and fit IDs make the derived space reproducible. In a real benchmark, "
        "fit centering parameters on training/reference proteins only."
    )
    return centered


def run_and_print_annotations(result: dict) -> dict:
    section("11. Categorical annotation analysis")
    annotations = [
        {
            "sequence_id": item["id"],
            "categorical": {"expected_group": item["metadata"]["expected_group"]},
        }
        for item in SEQUENCES
    ]
    analysis = analyze_categorical_annotations(
        result["models"],
        result["sequence_order"],
        annotations,
        AnnotationAnalysisSpec(
            feature="expected_group",
            representation=RepresentationMethod.MEAN,
            layer="model_default",
            k=(1, 2, 3),
            missing_policy=MissingAnnotationPolicy.ERROR,
            max_exact_sequences=100,
        ),
    )
    print(
        "S1/S2/S3 are labeled reference_like and S4 composition_control. This in-sample "
        "analysis asks whether cosine neighbors share those supplied labels; it is not a "
        "held-out accuracy estimate."
    )
    print(f"Class counts: {analysis['class_summary']}")
    for model_id, item in analysis["models"].items():
        print(f"\n{model_id}")
        for k, values in item["by_k"].items():
            print(
                f"  k={k}: micro={number(values['micro_consistency'])}, "
                f"macro={number(values['macro_consistency'])}, "
                f"chance={number(values['chance_baseline'])}, "
                f"chance-adjusted={number(values['chance_adjusted_consistency'])}, "
                f"mean recall={number(values['mean_recall_at_k'])}"
            )
    for warning in analysis["warnings"]:
        print(f"  warning: {warning}")
    print(f"Guardrail: {analysis['scientific_guardrail']}.")
    return analysis


def save_and_render(
    result: dict,
    diagnostics: dict,
    centered: dict,
    annotations: dict,
    output_root: Path,
) -> None:
    section("12. Numerical files and visualizations")
    store = ArtifactStore(output_root)
    descriptors = [
        store.save_json(JOB_ID, "comparison.json", result),
        store.save_json(JOB_ID, "diagnostics.json", diagnostics),
        store.save_json(JOB_ID, "mean_centered.json", centered),
        store.save_json(JOB_ID, "annotation_analysis.json", annotations),
    ]
    descriptors.extend(render_requested(result, store, job_id=JOB_ID))
    job_dir = output_root.resolve() / JOB_ID
    print(f"Output directory: {job_dir}")
    print(
        "JSON files contain full-precision numerical values and provenance. SVG files are "
        "dependency-free summaries; PCoA, UMAP, and dendrogram are interactive HTML files."
    )
    for descriptor in descriptors:
        print(
            f"  {descriptor['name']:<70} {human_bytes(descriptor['size']):>12}  "
            f"sha256={descriptor['sha256'][:12]}..."
        )
    warning_written = any(
        descriptor["name"] == "visualization_warnings.json" for descriptor in descriptors
    )
    warning_path = job_dir / "visualization_warnings.json"
    if warning_written:
        warnings = json.loads(warning_path.read_text(encoding="utf-8"))
        print("\nVisualization warnings (numerical analysis still completed):")
        for warning in warnings:
            print(f"  - {warning}")
    print(
        "\nInterpret plots as exploratory summaries. Four hand-selected proteins are useful "
        "for a software sanity check but are far too few for biological conclusions."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Common device supported by all three models (default: auto)",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "bf16"),
        default="auto",
        help="Common precision supported by all three models (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python_ex_output"),
        help="Artifact root directory (default: python_ex_output)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = build_request(device=args.device, precision=args.precision)
    pipeline = Pipeline()
    order = [item["id"] for item in SEQUENCES]

    section("Complete PLMs Framework example: ESM-2 8M, ESM-C 300M, and ProGen2 small")
    print(
        "This run exercises every implemented metric, all five representations supported by "
        "all three selected models, every cross-model method, diagnostics, mean-centering, "
        "categorical annotation analysis, and every visualization."
    )
    print(
        f"Requested execution: device={args.device}, precision={args.precision}, "
        "strict three-model completion=True"
    )
    print_sequence_design(order)
    print_estimate(pipeline.estimate(request))
    if not check_backends(pipeline):
        return 2

    section("Running the neural models")
    print("Downloading/loading checkpoints if necessary, then embedding the four sequences...")
    try:
        result = pipeline.compare(request)
    except FrameworkError as exc:
        print("\nThe strict comparison stopped with a structured framework error:")
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True))
        print(
            "Resolve that model/dependency/device problem and rerun. Strict mode is intentional "
            "so incomplete output is never presented as a three-model comparison."
        )
        return 1

    print(
        f"Completed in {result['provenance']['duration_seconds']:.3f} seconds; "
        f"successful models={list(result['models'])}; failures={len(result['failures'])}."
    )
    print_model_outputs(result)
    print_sequence_baselines(result)
    print_pooled_matrices(result)
    print_residue_results(result)
    print_cross_model_results(result)
    diagnostics = run_and_print_diagnostics(result)
    centered = run_and_print_transformation(result)
    annotations = run_and_print_annotations(result)
    save_and_render(result, diagnostics, centered, annotations, args.output_dir)

    section("Finished")
    print(
        "The exact duplicate, related variant, and unrelated control should now be visible in "
        "sequence, pooled-embedding, residue, cross-model, and visualization outputs. Review "
        "comparison.json for the complete machine-readable result and provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
