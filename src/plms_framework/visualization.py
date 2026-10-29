from __future__ import annotations

import hashlib
import html
import importlib.util
import math
import re

from .artifacts import ArtifactStore


def _first_matrix(result: dict) -> tuple[str, list[list[float]]] | None:
    for model_id, output in result.get("models", {}).items():
        for name, matrix in output.get("matrices", {}).items():
            return f"{model_id}: {name}", matrix
    for name, matrix in result.get("sequence_baselines", {}).items():
        return name, matrix
    return None


def render_requested(result: dict, store: ArtifactStore, job_id: str) -> list[dict]:
    artifacts = []
    warnings = []
    requested = result.get("requested_visualizations", [])
    matrix_item = _first_matrix(result)
    labels = result.get("sequence_order", [])
    for kind in requested:
        try:
            if kind == "clustered_heatmap" and matrix_item:
                title, matrix = matrix_item
                order = cluster_order(matrix, similarity=_is_similarity(title))
                clustered = [[matrix[i][j] for j in order] for i in order]
                clustered_labels = [labels[i] for i in order]
                artifacts.append(
                    store.save_text(
                        job_id,
                        "clustered_heatmap.svg",
                        heatmap_svg(clustered, clustered_labels, title),
                    )
                )
            elif kind == "distance_distribution" and matrix_item:
                title, matrix = matrix_item
                artifacts.append(
                    store.save_text(
                        job_id, "distance_distribution.svg", distribution_svg(matrix, title)
                    )
                )
            elif kind == "nearest_neighbor_graph" and matrix_item:
                title, matrix = matrix_item
                artifacts.append(
                    store.save_text(
                        job_id,
                        "nearest_neighbor_graph.svg",
                        graph_svg(matrix, labels, title, similarity=_is_similarity(title)),
                    )
                )
            elif kind == "model_agreement" and result.get("model_agreement", {}).get("spearman"):
                agreement = result["model_agreement"]
                artifacts.append(
                    store.save_text(
                        job_id,
                        "model_agreement.svg",
                        heatmap_svg(
                            agreement["spearman"],
                            agreement["model_order"],
                            "Model agreement (Spearman)",
                        ),
                    )
                )
            elif kind == "model_matrix_small_multiples":
                relationships = result.get("cross_model", {}).get("relationship_matrices", {})
                if not relationships:
                    warnings.append(
                        "model_matrix_small_multiples requires cross-model relationships"
                    )
                shared_order = (
                    cluster_order(next(iter(relationships.values())), similarity=True)
                    if relationships
                    else []
                )
                for model_id, matrix in relationships.items():
                    ordered = [[matrix[i][j] for j in shared_order] for i in shared_order]
                    ordered_labels = [labels[i] for i in shared_order]
                    artifacts.append(
                        store.save_text(
                            job_id,
                            f"model-matrix-{_artifact_slug(model_id)}.svg",
                            heatmap_svg(
                                ordered, ordered_labels, f"{model_id}: within-model cosine"
                            ),
                        )
                    )
            elif kind == "consensus_heatmap":
                consensus = (
                    result.get("cross_model", {}).get("comparisons", {}).get("consensus_rank", {})
                )
                if consensus.get("available"):
                    artifacts.append(
                        store.save_text(
                            job_id,
                            "consensus_heatmap.svg",
                            heatmap_svg(
                                consensus["values"],
                                consensus["sequence_order"],
                                "Cross-model rank consensus",
                            ),
                        )
                    )
                else:
                    warnings.append("consensus_heatmap requires an available consensus_rank result")
            elif kind == "model_disagreement":
                disagreement = (
                    result.get("cross_model", {})
                    .get("comparisons", {})
                    .get("pair_disagreement", {})
                )
                if disagreement.get("available"):
                    artifacts.append(
                        store.save_text(
                            job_id,
                            "model_disagreement.svg",
                            heatmap_svg(
                                disagreement["values"], labels, "Cross-model rank disagreement"
                            ),
                        )
                    )
                else:
                    warnings.append(
                        "model_disagreement requires an available pair_disagreement result"
                    )
            elif kind == "residue_model_agreement":
                comparisons = result.get("cross_model", {}).get("comparisons", {})
                residue_result = comparisons.get("residue_cka") or comparisons.get("residue_rsa")
                if not residue_result:
                    warnings.append("residue_model_agreement requires residue_cka or residue_rsa")
                else:
                    for sequence_id, agreement in residue_result.get("by_sequence", {}).items():
                        if not agreement.get("available"):
                            continue
                        artifacts.append(
                            store.save_text(
                                job_id,
                                f"residue-model-agreement-{_artifact_slug(sequence_id)}.svg",
                                heatmap_svg(
                                    agreement["values"],
                                    agreement["model_order"],
                                    f"Residue model agreement: {sequence_id}",
                                ),
                            )
                        )
            elif kind == "residue_similarity_map":
                saved = False
                for model_id, output in result.get("models", {}).items():
                    for pair, pair_result in output.get("residue_pairs", {}).items():
                        if "similarity_matrix" in pair_result:
                            safe_model = _artifact_slug(model_id)[:40]
                            safe_pair = _artifact_slug(pair.replace("::", "-"))[:80]
                            identity = hashlib.sha256(
                                repr((model_id, pair)).encode("utf-8")
                            ).hexdigest()[:12]
                            artifacts.append(
                                store.save_text(
                                    job_id,
                                    f"residue-{safe_model}-{safe_pair}-{identity}.svg",
                                    heatmap_svg(
                                        pair_result["similarity_matrix"],
                                        [],
                                        f"{model_id}: residue similarity: {pair}",
                                    ),
                                )
                            )
                            saved = True
                if not saved:
                    warnings.append(
                        "residue_similarity_map requires residue representation and a residue metric"
                    )
            elif kind in {"pcoa", "umap", "dendrogram"}:
                if not matrix_item:
                    warnings.append(f"{kind} requires a pairwise matrix")
                elif (
                    importlib.util.find_spec("plotly") is None
                    or importlib.util.find_spec("numpy") is None
                ):
                    warnings.append(
                        f"{kind} requires the visual extra: pip install 'plms-framework[visual]'"
                    )
                else:
                    title, matrix = matrix_item
                    document = advanced_plot(kind, matrix, labels, title)
                    artifacts.append(store.save_text(job_id, f"{kind}.html", document))
            else:
                warnings.append(f"Could not render {kind}: no compatible data")
        # Optional plotting libraries may raise backend-specific exception types.
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not render {kind}: {exc}")
    if warnings:
        artifacts.append(store.save_json(job_id, "visualization_warnings.json", warnings))
    return artifacts


def heatmap_svg(matrix: list[list[float]], labels: list[str], title: str) -> str:
    n = len(matrix)
    columns = len(matrix[0]) if matrix else 0
    cell = max(4, min(36, 600 // max(1, n, columns)))
    margin = 160 if labels and n <= 100 else 40
    size = cell * n
    plot_width = cell * columns
    values = [value for row in matrix for value in row]
    low, high = min(values), max(values)
    span = high - low or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width + margin + 20}" height="{size + margin + 40}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="10" y="20" font-family="sans-serif" font-size="14">{html.escape(title)}</text>',
    ]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ratio = (value - low) / span
            red, blue = int(245 * ratio), int(245 * (1 - ratio))
            color = f"rgb({red},70,{blue})"
            parts.append(
                f'<rect x="{margin + j * cell}" y="{30 + i * cell}" width="{cell}" height="{cell}" fill="{color}"><title>{value:.6g}</title></rect>'
            )
    if labels and n <= 100:
        for index, label in enumerate(labels):
            escaped = html.escape(str(label))
            y = 30 + index * cell + cell * 0.72
            parts.append(
                f'<text x="{margin - 5}" y="{y}" text-anchor="end" font-family="sans-serif" font-size="9">{escaped}</text>'
            )
            x = margin + index * cell + cell * 0.5
            parts.append(
                f'<text x="{x}" y="{size + 38}" transform="rotate(60 {x} {size + 38})" font-family="sans-serif" font-size="9">{escaped}</text>'
            )
    parts.append(
        f'<text x="10" y="{size + 35}" font-family="sans-serif" font-size="10">min={low:.5g}, max={high:.5g}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _is_similarity(title: str) -> bool:
    return any(
        word in title.lower() for word in ("similarity", "identity", "jaccard", "ratio", "blosum")
    )


def _artifact_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return slug[:100] or "item"


def cluster_order(matrix: list[list[float]], *, similarity: bool) -> list[int]:
    """Deterministic average-linkage leaf ordering without optional dependencies."""
    clusters: list[list[int]] = [[index] for index in range(len(matrix))]
    if len(clusters) < 2:
        return [0] if clusters else []

    def distance(left: list[int], right: list[int]) -> float:
        values = [matrix[i][j] for i in left for j in right]
        mean = sum(values) / len(values)
        return 1.0 - mean if similarity else mean

    while len(clusters) > 1:
        pair = min(
            (
                (distance(clusters[i], clusters[j]), i, j)
                for i in range(len(clusters))
                for j in range(i + 1, len(clusters))
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        _, i, j = pair
        left, right = clusters[i], clusters[j]
        orientations = (
            left + right,
            left + list(reversed(right)),
            list(reversed(left)) + right,
            list(reversed(left)) + list(reversed(right)),
        )
        merged = min(
            orientations,
            key=lambda order: (
                1.0 - matrix[order[len(left) - 1]][order[len(left)]]
                if similarity
                else matrix[order[len(left) - 1]][order[len(left)]],
                order,
            ),
        )
        clusters[i] = merged
        clusters.pop(j)
    return clusters[0]


def distribution_svg(matrix: list[list[float]], title: str) -> str:
    values = [matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))]
    if not values:
        values = [0.0]
    low, high = min(values), max(values)
    bins = [0] * 20
    span = high - low or 1.0
    for value in values:
        bins[min(19, int((value - low) / span * 20))] += 1
    maximum = max(bins) or 1
    bars = []
    for i, count in enumerate(bins):
        height = 180 * count / maximum
        bars.append(
            f'<rect x="{45 + i * 24}" y="{225 - height}" width="20" height="{height}" fill="#4169a1"><title>{count}</title></rect>'
        )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="550" height="270"><rect width="100%" height="100%" fill="white"/><text x="20" y="20" font-family="sans-serif">{html.escape(title)}</text>{"".join(bars)}<text x="45" y="250" font-family="sans-serif" font-size="11">{low:.4g}</text><text x="500" y="250" font-family="sans-serif" font-size="11">{high:.4g}</text></svg>'


def graph_svg(
    matrix: list[list[float]], labels: list[str], title: str, k: int = 3, similarity: bool = True
) -> str:
    n = len(matrix)
    center, radius = 310, 240
    positions = [
        (
            center + radius * math.cos(2 * math.pi * i / n),
            center + radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    edges = set()
    for i, row in enumerate(matrix):
        neighbors = sorted(
            (j for j in range(n) if j != i), key=lambda j: row[j], reverse=similarity
        )[: min(k, n - 1)]
        edges.update(tuple(sorted((i, j))) for j in neighbors)
    lines = [
        f'<line x1="{positions[i][0]:.1f}" y1="{positions[i][1]:.1f}" x2="{positions[j][0]:.1f}" y2="{positions[j][1]:.1f}" stroke="#aaa"/>'
        for i, j in edges
    ]
    nodes = []
    for i, (x, y) in enumerate(positions):
        label = html.escape(labels[i] if i < len(labels) else str(i))
        nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#b43c5a"><title>{label}</title></circle><text x="{x + 9:.1f}" y="{y + 3:.1f}" font-family="sans-serif" font-size="9">{label}</text>'
        )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="700" height="640"><rect width="100%" height="100%" fill="white"/><text x="10" y="20" font-family="sans-serif">{html.escape(title)} — top-{k} graph</text>{"".join(lines)}{"".join(nodes)}</svg>'


def advanced_plot(kind: str, matrix: list[list[float]], labels: list[str], title: str) -> str:
    import numpy as np
    import plotly.express as px

    data = np.asarray(matrix, dtype=float)
    distance = 1.0 - data if _is_similarity(title) else data
    if kind == "pcoa":
        n = len(distance)
        centering = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * centering @ (distance**2) @ centering
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        coordinates = vectors[:, :2] * np.sqrt(np.maximum(values[:2], 0))
        figure = px.scatter(
            x=coordinates[:, 0], y=coordinates[:, 1], text=labels, title=f"PCoA — {title}"
        )
    elif kind == "umap":
        import umap

        coordinates = umap.UMAP(metric="precomputed", random_state=42).fit_transform(distance)
        figure = px.scatter(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            text=labels,
            title=f"UMAP (exploratory) — {title}",
        )
    else:
        import plotly.figure_factory as ff
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        linkage_matrix = linkage(squareform(distance, checks=False), method="average")
        figure = ff.create_dendrogram(data, labels=labels, linkagefun=lambda _: linkage_matrix)
        figure.update_layout(title=f"Dendrogram — {title}")
    return figure.to_html(include_plotlyjs="cdn", full_html=True)
