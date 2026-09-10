import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from physics_engine.dock_vina import (
    batch_dock_library,
    ensure_vina_binary,
    parse_vina_score,
    prepare_ligand_pdbqt,
)
from run_pipeline import is_stale
from vector_engine.predict_bbb import run_bbb_inference


def test_is_stale_logic(tmp_path: Path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    # Non-existent output is stale
    input_file.write_text("dummy")
    assert is_stale([input_file], output_file) is True

    # Output newer than input is fresh
    output_file.write_text("result")
    assert is_stale([input_file], output_file) is False


def test_parse_vina_score_extraction():
    sample_stdout = """
-----+------------+----------+----------
   1       -8.452      0.000      0.000
   2       -7.100      1.230      2.100
"""
    assert parse_vina_score(sample_stdout) == -8.452
    assert parse_vina_score("No valid modes detected") is None


def test_prepare_ligand_pdbqt_invalid_smiles(tmp_path: Path):
    target_path = tmp_path / "invalid.pdbqt"
    assert prepare_ligand_pdbqt("INVALID_STRING_123", target_path) is False
    assert not target_path.exists()


@patch("urllib.request.urlretrieve")
@patch("platform.system")
def test_ensure_vina_binary_posix(mock_system, mock_retrieve, tmp_path: Path):
    mock_system.return_value = "Linux"
    bin_dir = tmp_path / "bin"
    expected_bin = bin_dir / "vina"

    def side_effect(url, dest):
        Path(dest).write_text("binary_stub")

    mock_retrieve.side_effect = side_effect

    res = ensure_vina_binary(bin_dir)
    assert res == expected_bin
    assert res.exists()