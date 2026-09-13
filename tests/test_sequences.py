import unittest

from plms_framework.errors import ValidationError
from plms_framework.models import ProteinSequence, Region
from plms_framework.sequences import canonicalize_many, parse_fasta


class SequenceTests(unittest.TestCase):
    def test_canonicalization(self):
        result = canonicalize_many((ProteinSequence(" p1 ", " mk\nt "),), "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(result[0].id, "p1")
        self.assertEqual(result[0].sequence, "MKT")

    def test_invalid_residue(self):
        with self.assertRaises(ValidationError):
            canonicalize_many((ProteinSequence("p1", "MK*"),), "ACDEFGHIKLMNPQRSTVWY")

    def test_duplicate_ids(self):
        with self.assertRaises(ValidationError):
            canonicalize_many((ProteinSequence("p", "M"), ProteinSequence("p", "K")), "MK")

    def test_region_coordinates_are_one_based_inclusive(self):
        canonicalize_many((ProteinSequence("p", "MKT", regions=(Region("r", 1, 3),)),), "MKT")
        with self.assertRaises(ValidationError):
            canonicalize_many((ProteinSequence("p", "MKT", regions=(Region("r", 0, 2),)),), "MKT")

    def test_fasta(self):
        records = parse_fasta(">a\nMKT\n>second protein\nAAA\n")
        self.assertEqual(
            [(r.id, r.sequence) for r in records], [("a", "MKT"), ("second protein", "AAA")]
        )


if __name__ == "__main__":
    unittest.main()
