from pathlib import Path
from target.prepare_target import compute_native_centroid, validate_receptor_integrity

def test_centroid_computation(tmp_path: Path):
    mock_pdb = tmp_path / "mock.pdb"
    mock_pdb.write_text("HETATM    1  C   HUX A   1      10.000  20.000  -5.000  1.00  0.00           C  \n"
                        "HETATM    2  O   HUX A   1      12.000  18.000  -3.000  1.00  0.00           O  \n")
    cx, cy, cz = compute_native_centroid(mock_pdb, {"HUX"})
    assert cx == 11.0 and cy == 19.0 and cz == -4.0
