from __future__ import annotations

import math
from collections.abc import Callable

from .errors import ValidationError


def _check_vectors(left: list[float], right: list[float]) -> None:
    if not left or len(left) != len(right):
        raise ValidationError("Vectors must be non-empty and have the same dimension")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    _check_vectors(left, right)
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - cosine_similarity(left, right)


def angular_distance(left: list[float], right: list[float]) -> float:
    return math.acos(cosine_similarity(left, right)) / math.pi


def euclidean_distance(left: list[float], right: list[float]) -> float:
    _check_vectors(left, right)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def correlation_distance(left: list[float], right: list[float]) -> float:
    _check_vectors(left, right)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    return 1.0 - cosine_similarity(
        [value - left_mean for value in left], [value - right_mean for value in right]
    )


VECTOR_METRICS: dict[str, Callable[[list[float], list[float]], float]] = {
    "cosine_similarity": cosine_similarity,
    "cosine_distance": cosine_distance,
    "angular_distance": angular_distance,
    "euclidean_distance": euclidean_distance,
    "correlation_distance": correlation_distance,
}


def pairwise(vectors: list[list[float]], metric: str) -> list[list[float]]:
    try:
        function = VECTOR_METRICS[metric]
    except KeyError as exc:
        raise ValidationError(f"Unknown vector metric: {metric}") from exc
    size = len(vectors)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = function(vectors[i], vectors[j])
            matrix[i][j] = matrix[j][i] = value
    return matrix


def length_ratio(left: str, right: str) -> float:
    return min(len(left), len(right)) / max(len(left), len(right))


def kmer_jaccard(left: str, right: str, k: int = 3) -> float:
    if k <= 0:
        raise ValidationError("k must be positive")
    left_set = {left[i : i + k] for i in range(max(0, len(left) - k + 1))}
    right_set = {right[i : i + k] for i in range(max(0, len(right) - k + 1))}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else float(left == right)


def global_sequence_identity(left: str, right: str) -> float:
    """Needleman-Wunsch identity using deterministic match/mismatch/gap scores."""
    m, n = len(left), len(right)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    trace = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        score[i][0], trace[i][0] = -i, "U"
    for j in range(1, n + 1):
        score[0][j], trace[0][j] = -j, "L"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choices = (
                (score[i - 1][j - 1] + (1 if left[i - 1] == right[j - 1] else -1), "D"),
                (score[i - 1][j] - 1, "U"),
                (score[i][j - 1] - 1, "L"),
            )
            score[i][j], trace[i][j] = max(choices, key=lambda item: (item[0], item[1] == "D"))
    i, j, matches, columns = m, n, 0, 0
    while i or j:
        direction = trace[i][j]
        if direction == "D":
            matches += left[i - 1] == right[j - 1]
            i -= 1
            j -= 1
        elif direction == "U":
            i -= 1
        else:
            j -= 1
        columns += 1
    return matches / columns if columns else 1.0


SEQUENCE_METRICS: dict[str, Callable[[str, str], float]] = {
    "length_ratio": length_ratio,
    "sequence_identity": global_sequence_identity,
    "kmer_jaccard": kmer_jaccard,
}

_BLOSUM62_TEXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1
"""


def _parse_blosum62() -> dict[tuple[str, str], int]:
    lines = [line.split() for line in _BLOSUM62_TEXT.strip().splitlines()]
    header = lines[0]
    return {
        (row[0], amino): int(value)
        for row in lines[1:]
        for amino, value in zip(header, row[1:], strict=True)
    }


BLOSUM62 = _parse_blosum62()


def _blosum_value(left: str, right: str) -> int:
    return BLOSUM62.get((left, right), BLOSUM62.get(("X", right), -1))


def _blosum_alignment_score(left: str, right: str, *, local: bool, gap: int = -4) -> int:
    previous = [0 if local else j * gap for j in range(len(right) + 1)]
    best = 0
    for i, a in enumerate(left, start=1):
        current = [0 if local else i * gap] + [0] * len(right)
        for j, b in enumerate(right, start=1):
            options = (
                previous[j - 1] + _blosum_value(a, b),
                previous[j] + gap,
                current[j - 1] + gap,
            )
            current[j] = max((0, *options)) if local else max(options)
            best = max(best, current[j])
        previous = current
    return best if local else previous[-1]


def normalized_blosum62(left: str, right: str, *, local: bool) -> float:
    raw = _blosum_alignment_score(left, right, local=local)
    self_left = _blosum_alignment_score(left, left, local=local)
    self_right = _blosum_alignment_score(right, right, local=local)
    denominator = math.sqrt(max(1, self_left) * max(1, self_right))
    return raw / denominator


SEQUENCE_METRICS.update(
    {
        "blosum62_global": lambda left, right: normalized_blosum62(left, right, local=False),
        "blosum62_local": lambda left, right: normalized_blosum62(left, right, local=True),
    }
)


def pairwise_sequences(sequences: list[str], metric: str) -> list[list[float]]:
    function = SEQUENCE_METRICS[metric]
    size = len(sequences)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            matrix[i][j] = matrix[j][i] = function(sequences[i], sequences[j])
    return matrix


def residue_similarity(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[cosine_similarity(a, b) for b in right] for a in left]


def align_similarity(
    similarity: list[list[float]], *, local: bool, gap_penalty: float = -0.5
) -> dict:
    if not similarity or not similarity[0]:
        raise ValidationError("Residue similarity matrix cannot be empty")
    m, n = len(similarity), len(similarity[0])
    scores = [[0.0] * (n + 1) for _ in range(m + 1)]
    trace = [[""] * (n + 1) for _ in range(m + 1)]
    if not local:
        for i in range(1, m + 1):
            scores[i][0], trace[i][0] = i * gap_penalty, "U"
        for j in range(1, n + 1):
            scores[0][j], trace[0][j] = j * gap_penalty, "L"
    best = (0.0, 0, 0)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choices = [
                (scores[i - 1][j - 1] + similarity[i - 1][j - 1], "D"),
                (scores[i - 1][j] + gap_penalty, "U"),
                (scores[i][j - 1] + gap_penalty, "L"),
            ]
            if local:
                choices.append((0.0, "0"))
            scores[i][j], trace[i][j] = max(choices, key=lambda item: (item[0], item[1] == "D"))
            if scores[i][j] > best[0]:
                best = (scores[i][j], i, j)
    if local:
        raw_score, i, j = best
    else:
        raw_score, i, j = scores[m][n], m, n
    path: list[tuple[int | None, int | None]] = []
    matched_scores: list[float] = []
    while i or j:
        if local and (scores[i][j] <= 0 or trace[i][j] == "0"):
            break
        direction = trace[i][j]
        if direction == "D":
            path.append((i, j))
            matched_scores.append(similarity[i - 1][j - 1])
            i -= 1
            j -= 1
        elif direction == "U":
            path.append((i, None))
            i -= 1
        elif direction == "L":
            path.append((None, j))
            j -= 1
        else:
            break
    path.reverse()
    return {
        "raw_score": raw_score,
        "normalized_score": raw_score / max(1, len(path)),
        "mean_matched_similarity": sum(matched_scores) / len(matched_scores)
        if matched_scores
        else 0.0,
        "gap_penalty": gap_penalty,
        "path": path,
    }


def symmetric_best_match(similarity: list[list[float]]) -> float:
    rows = sum(max(row) for row in similarity) / len(similarity)
    columns = sum(
        max(similarity[i][j] for i in range(len(similarity))) for j in range(len(similarity[0]))
    )
    columns /= len(similarity[0])
    return (rows + columns) / 2


def sinkhorn_similarity(
    similarity: list[list[float]], *, epsilon: float = 0.1, iterations: int = 100
) -> dict:
    """Balanced entropy-regularized OT over uniform residue mass."""
    if epsilon <= 0 or iterations <= 0:
        raise ValidationError("Sinkhorn epsilon and iterations must be positive")
    m, n = len(similarity), len(similarity[0])
    kernel = [[math.exp(-(1.0 - value) / epsilon) for value in row] for row in similarity]
    left_mass, right_mass = [1.0 / m] * m, [1.0 / n] * n
    u, v = [1.0] * m, [1.0] * n
    tiny = 1e-300
    for _ in range(iterations):
        u = [left_mass[i] / max(tiny, sum(kernel[i][j] * v[j] for j in range(n))) for i in range(m)]
        v = [
            right_mass[j] / max(tiny, sum(kernel[i][j] * u[i] for i in range(m))) for j in range(n)
        ]
    transport_cost = 0.0
    transported_mass = 0.0
    for i in range(m):
        for j in range(n):
            mass = u[i] * kernel[i][j] * v[j]
            transported_mass += mass
            transport_cost += mass * (1.0 - similarity[i][j])
    return {
        "similarity": 1.0 - transport_cost / max(tiny, transported_mass),
        "transport_cost": transport_cost,
        "transported_mass": transported_mass,
        "epsilon": epsilon,
        "iterations": iterations,
        "method": "balanced_entropic_sinkhorn_uniform_mass",
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def upper_triangle(matrix: list[list[float]]) -> list[float]:
    return [matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))]


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValidationError("Correlation requires equally sized non-empty inputs")
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    a = [v - left_mean for v in left]
    b = [v - right_mean for v in right]
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator if denominator else 0.0


def matrix_spearman(left: list[list[float]], right: list[list[float]]) -> float:
    return pearson(_rank(upper_triangle(left)), _rank(upper_triangle(right)))
