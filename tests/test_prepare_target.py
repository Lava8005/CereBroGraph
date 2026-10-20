"""TDD Suite for Receptor Integrity, Centroid Precision, and State Caching."""
import json
from pathlib import Path
from target.prepare_target import compute_native_centroid, validate_receptor_integrity

def test_centroid_excludes_waters(tmp_path: Path):
    """Ensure the centroid math strictly ignores hydration shell coordinates."""
    mock_pdb = tmp_path / "mock.pdb"
    mock_pdb.write_text(
        "HETATM    1  C   HUX A   1      10.000  20.000  -5.000  1.00  0.00           C  \n"
        "HETATM    2  O   HUX A   1      12.000  18.000  -3.000  1.00  0.00           O  \n"
        "HETATM    3  O   HOH A   2      99.000  99.000  99.000  1.00  0.00           O  \n" # MUST be ignored
    )
    # Passed ONLY the native ligand
    cx, cy, cz = compute_native_centroid(mock_pdb, {"HUX"})
    assert cx == 11.0 and cy == 19.0 and cz == -4.0

def test_empty_pdbqt_rejection(tmp_path: Path):
    """Ensure failed obabel conversions trigger a pipeline halt."""
    empty_file = tmp_path / "failed_receptor.pdbqt"
    empty_file.write_text("")
    assert validate_receptor_integrity(empty_file) is False