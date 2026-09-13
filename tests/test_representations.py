import math
import unittest

from plms_framework.models import (
    Normalization,
    ProteinSequence,
    RawEmbedding,
    Region,
    RepresentationMethod,
    RepresentationSpec,
)
from plms_framework.representations import materialize, mean_pool


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.raw = RawEmbedding("p", [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
        self.sequence = ProteinSequence("p", "MKT", regions=(Region("tail", 2, 3),))

    def test_mean_pool(self):
        self.assertEqual(mean_pool(self.raw.residues), [2 / 3, 4 / 3])

    def test_mean_is_l2_normalized(self):
        result = materialize(self.raw, self.sequence, RepresentationSpec())
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in result.values)), 1.0)
        self.assertAlmostEqual(result.norm_before, math.sqrt(20) / 3)

    def test_residue_none_preserves_rows(self):
        spec = RepresentationSpec(RepresentationMethod.RESIDUE, normalize=Normalization.NONE)
        self.assertEqual(materialize(self.raw, self.sequence, spec).values, self.raw.residues)

    def test_region_mean(self):
        spec = RepresentationSpec(RepresentationMethod.REGION_MEAN, normalize=Normalization.NONE)
        result = materialize(self.raw, self.sequence, spec)
        self.assertEqual(result.values[0]["values"], [0.5, 2.0])

    def test_windows(self):
        spec = RepresentationSpec(
            RepresentationMethod.WINDOW_MEAN,
            normalize=Normalization.NONE,
            window_size=2,
            window_step=1,
        )
        result = materialize(self.raw, self.sequence, spec)
        self.assertEqual(result.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
