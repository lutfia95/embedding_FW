import unittest

from plms_framework.annotations import analyze_categorical_annotations
from plms_framework.errors import ValidationError
from plms_framework.external import build_imported_space
from plms_framework.models import AnnotationAnalysisSpec, MissingAnnotationPolicy


class AnnotationAnalysisTests(unittest.TestCase):
    def setUp(self):
        imported = build_imported_space(
            space_name="label-fixture",
            representations=[
                {
                    "method": "mean",
                    "embeddings": [
                        {"sequence_id": "a1", "values": [1.0, 0.0]},
                        {"sequence_id": "a2", "values": [0.9, 0.1]},
                        {"sequence_id": "b1", "values": [-1.0, 0.0]},
                        {"sequence_id": "b2", "values": [-0.9, -0.1]},
                    ],
                }
            ],
        )
        self.outputs = imported["models"]
        self.order = imported["sequence_order"]
        self.annotations = [
            {"sequence_id": sequence_id, "categorical": {"family": sequence_id[0].upper()}}
            for sequence_id in self.order
        ]

    def test_perfect_knn_label_consistency(self):
        result = analyze_categorical_annotations(
            self.outputs,
            self.order,
            self.annotations,
            AnnotationAnalysisSpec(feature="family", k=(1,)),
        )
        model = result["models"]["label-fixture"]
        scores = model["by_k"]["1"]
        self.assertTrue(result["available"])
        self.assertEqual(scores["micro_consistency"], 1.0)
        self.assertEqual(scores["macro_consistency"], 1.0)
        self.assertEqual(scores["precision_at_k"], 1.0)
        self.assertEqual(scores["mean_recall_at_k"], 1.0)
        self.assertAlmostEqual(scores["chance_baseline"], 1 / 3)
        self.assertEqual(scores["chance_adjusted_consistency"], 1.0)
        self.assertEqual(scores["class_mixing"]["counts"], [[2, 0], [0, 2]])

    def test_drop_missing_annotations(self):
        result = analyze_categorical_annotations(
            self.outputs,
            self.order,
            self.annotations[:-1],
            AnnotationAnalysisSpec(
                feature="family",
                k=(1,),
                missing_policy=MissingAnnotationPolicy.DROP,
            ),
        )
        self.assertEqual(result["analyzed_sequence_count"], 3)
        self.assertEqual(result["dropped_sequence_ids"], ["b2"])

    def test_missing_annotations_error_by_default(self):
        with self.assertRaisesRegex(ValidationError, "Missing categorical feature"):
            analyze_categorical_annotations(
                self.outputs,
                self.order,
                self.annotations[:-1],
                AnnotationAnalysisSpec(feature="family", k=(1,)),
            )

    def test_single_class_is_rejected(self):
        annotations = [
            {"sequence_id": sequence_id, "categorical": {"family": "same"}}
            for sequence_id in self.order
        ]
        with self.assertRaisesRegex(ValidationError, "at least two classes"):
            analyze_categorical_annotations(
                self.outputs,
                self.order,
                annotations,
                AnnotationAnalysisSpec(feature="family", k=(1,)),
            )


if __name__ == "__main__":
    unittest.main()
