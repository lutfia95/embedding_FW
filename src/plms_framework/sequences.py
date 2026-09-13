from __future__ import annotations

import hashlib
import re

from .errors import ValidationError
from .models import ProteinSequence

_FORMATTING_WHITESPACE = re.compile(r"\s+")


def canonicalize(item: ProteinSequence, alphabet: str) -> ProteinSequence:
    identifier = item.id.strip()
    if not identifier:
        raise ValidationError("Sequence ID cannot be empty")
    sequence = _FORMATTING_WHITESPACE.sub("", item.sequence).upper()
    if not sequence:
        raise ValidationError(f"Sequence {identifier!r} is empty")
    invalid = sorted(set(sequence).difference(alphabet))
    if invalid:
        raise ValidationError(
            f"Sequence {identifier!r} contains unsupported residues: {', '.join(invalid)}",
            context={"sequence_id": identifier, "invalid_residues": invalid},
        )
    for region in item.regions:
        region.validate(len(sequence))
    return ProteinSequence(identifier, sequence, dict(item.metadata), tuple(item.regions))


def canonicalize_many(
    items: tuple[ProteinSequence, ...], alphabet: str
) -> tuple[ProteinSequence, ...]:
    if not items:
        raise ValidationError("At least one sequence is required")
    canonical = tuple(canonicalize(item, alphabet) for item in items)
    ids = [item.id for item in canonical]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValidationError("Duplicate sequence IDs are not allowed", context={"ids": duplicates})
    return canonical


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def parse_fasta(text: str) -> tuple[ProteinSequence, ...]:
    records: list[ProteinSequence] = []
    identifier: str | None = None
    chunks: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append(ProteinSequence(identifier, "".join(chunks)))
            identifier = line[1:].strip()
            if not identifier:
                raise ValidationError(f"Empty FASTA identifier at line {line_number}")
            chunks = []
        elif identifier is None:
            raise ValidationError(
                f"FASTA sequence appears before first header at line {line_number}"
            )
        else:
            chunks.append(line)
    if identifier is not None:
        records.append(ProteinSequence(identifier, "".join(chunks)))
    if not records:
        raise ValidationError("No FASTA records found")
    return tuple(records)
