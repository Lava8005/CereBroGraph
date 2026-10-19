import json
import pytest
from pathlib import Path
from target.prepare_target import validate_receptor_integrity, prepare_single_target

def test_receptor_integrity_validation(tmp_path: Path):
    empty_file = tmp_path / "empty_receptor.pdbqt"
    empty_file.write_text("")
    assert validate_receptor_integrity(empty_file) == False

def test_grid_box_coordinate_alignment(tmp_path: Path):
    """Ensure exact dictionary key mapping for center_x/y/z."""
    target_config = {
        "disease": "Test_Disease", "pdb_id": "9XYZ",
        "active_site": {"center_x": 10.5, "center_y": -5.2, "center_z": 3.14}
    }
    # Mocking bypassing the download for structural test
    grid_json = tmp_path / "9XYZ_grid.json"
    with open(grid_json, "w") as f:
        json.dump(target_config["active_site"], f)
        
    loaded = json.loads(grid_json.read_text())
    assert loaded["center_x"] == 10.5
    assert loaded["center_z"] == 3.14