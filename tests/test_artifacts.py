import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from plms_framework.artifacts import ArtifactStore
from plms_framework.errors import ValidationError
from plms_framework.visualization import cluster_order, heatmap_svg, render_requested


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

    def test_rectangular_residue_map_fits_inside_svg(self):
        for rows, columns in ((2, 12), (12, 2)):
            with self.subTest(rows=rows, columns=columns):
                document = heatmap_svg([[0.5] * columns for _ in range(rows)], [], "Residues")
                root = ET.fromstring(document)
                width, height = float(root.attrib["width"]), float(root.attrib["height"])
                cells = [node for node in root if "x" in node.attrib and "width" in node.attrib]
                self.assertEqual(len(cells), rows * columns)
                for cell in cells:
                    self.assertLessEqual(
                        float(cell.attrib["x"]) + float(cell.attrib["width"]), width
                    )
                    self.assertLessEqual(
                        float(cell.attrib["y"]) + float(cell.attrib["height"]), height
                    )

    def test_residue_maps_keep_every_model_and_pair(self):
        # These IDs produce the same readable filename slug; they must stay distinct.
        result = {
            "requested_visualizations": ["residue_similarity_map"],
            "models": {
                model: {
                    "residue_pairs": {
                        pair: {"similarity_matrix": [[value, value + 0.1]]}
                        for pair, value in (("a/b::c", 0.2), ("a:b::c", 0.5))
                    }
                }
                for model in ("model/a", "model:a")
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifacts = render_requested(result, store, "residue-test")
            self.assertEqual(len(artifacts), 4)
            self.assertEqual(len({item["name"] for item in artifacts}), 4)
            saved = {item["name"]: item["sha256"] for item in store.list("residue-test")}
            self.assertEqual(saved, {item["name"]: item["sha256"] for item in artifacts})
            documents = [store.path("residue-test", item["name"]).read_text() for item in artifacts]
            for model in result["models"]:
                self.assertEqual(sum(model in document for document in documents), 2)

    def test_cluster_order_is_a_permutation(self):
        matrix = [[1.0, 0.1, 0.9], [0.1, 1.0, 0.2], [0.9, 0.2, 1.0]]
        order = cluster_order(matrix, similarity=True)
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertEqual(abs(order.index(0) - order.index(2)), 1)


if __name__ == "__main__":
    unittest.main()
