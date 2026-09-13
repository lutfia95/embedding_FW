import math
import unittest

from plms_framework.metrics import (
    align_similarity,
    angular_distance,
    cosine_similarity,
    euclidean_distance,
    global_sequence_identity,
    kmer_jaccard,
    matrix_spearman,
    normalized_blosum62,
    residue_similarity,
    sinkhorn_similarity,
)


class MetricTests(unittest.TestCase):
    def test_cosine_and_angular(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(angular_distance([1, 0], [0, 1]), 0.5)

    def test_normalized_euclidean_relation(self):
        left = [1.0, 0.0]
        right = [math.sqrt(0.5), math.sqrt(0.5)]
        self.assertAlmostEqual(
            euclidean_distance(left, right) ** 2, 2 - 2 * cosine_similarity(left, right)
        )

    def test_identity_and_kmer(self):
        self.assertEqual(global_sequence_identity("MKT", "MKT"), 1.0)
        self.assertGreater(global_sequence_identity("MKT", "MAT"), 0.5)
        self.assertEqual(kmer_jaccard("ABCDE", "ABCDE"), 1.0)

    def test_blosum_self_is_one(self):
        self.assertAlmostEqual(normalized_blosum62("MKT", "MKT", local=False), 1.0)
        self.assertAlmostEqual(normalized_blosum62("MKT", "MKT", local=True), 1.0)

    def test_residue_alignment(self):
        similarity = residue_similarity([[1, 0], [0, 1]], [[1, 0], [0, 1]])
        global_result = align_similarity(similarity, local=False)
        self.assertEqual(global_result["path"], [(1, 1), (2, 2)])
        self.assertAlmostEqual(global_result["normalized_score"], 1.0)

    def test_sinkhorn(self):
        result = sinkhorn_similarity([[1.0, 0.0], [0.0, 1.0]], iterations=50)
        self.assertGreater(result["similarity"], 0.9)
        self.assertAlmostEqual(result["transported_mass"], 1.0, places=6)

    def test_matrix_spearman(self):
        left = [[1, 0.8, 0.2], [0.8, 1, 0.5], [0.2, 0.5, 1]]
        self.assertAlmostEqual(matrix_spearman(left, left), 1.0)


if __name__ == "__main__":
    unittest.main()
