# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bioemu.utils import parse_sequence


@pytest.fixture
def sequence() -> str:
    # Needs to be long enough to trigger OSError in parse_sequence.
    return "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def seq_record(sequence: str) -> SeqRecord:
    return SeqRecord(Seq(sequence), id="test", description="test")


@pytest.fixture
def tmp_fasta_file(tmp_path: Path, seq_record: SeqRecord) -> str:
    fasta_file_path = tmp_path / "test.fasta"
    SeqIO.write(seq_record, fasta_file_path, "fasta")
    return str(fasta_file_path)


def test_parse_sequence(sequence: str) -> None:
    assert parse_sequence(sequence) == sequence


def test_parse_sequence_from_file(tmp_fasta_file: str, sequence: str) -> None:
    assert parse_sequence(tmp_fasta_file) == sequence
