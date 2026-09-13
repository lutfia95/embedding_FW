import tempfile
import unittest
from pathlib import Path

from plms_framework.artifacts import ArtifactStore
from plms_framework.errors import ValidationError
from plms_framework.visualization import cluster_order, heatmap_svg


class ArtifactTests(unittest.TestCase):
    def test_atomic_json_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            descriptor = store.save_json("job1", "result.json", {"ok": True})
            self.assertEqual(len(descriptor["sha256"]), 64)
            self.assertTrue((Path(directory) / "job1" / "result.json").exists())

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValidationError):
            ArtifactStore(directory).path("job", "../secret")

    def test_svg_escapes_labels(self):
        document = heatmap_svg([[1.0]], ["<protein>"], "title")
        self.assertIn("&lt;protein&gt;", document)
        self.assertNotIn("> <protein>", document)

    def test_cluster_order_is_a_permutation(self):
        matrix = [[1.0, 0.1, 0.9], [0.1, 1.0, 0.2], [0.9, 0.2, 1.0]]
        order = cluster_order(matrix, similarity=True)
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(abs(order.index(0) - order.index(2)), 1)


if __name__ == "__main__":
    unittest.main()
