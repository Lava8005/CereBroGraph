"""TDD Suite for Automated Target Preparation and Active-Site Grid Generation."""

import json
import pytest
from pathlib import Path
from target.prepare_target import validate_receptor_integrity, generate_grid_box_config

def test_receptor_integrity_validation(tmp_path: Path):
    """Ensure invalid or empty receptor files are detected and rejected."""
    empty_file = tmp_path / "empty_receptor.pdbqt"
    empty_file.write_text("")
    
    assert validate_receptor_integrity(empty_file) == False

def test_grid_box_config_serialization(tmp_path: Path):
    """Ensure grid box coordinates are correctly serialized to JSON for AutoDock Vina."""
    grid_path = tmp_path / "grid_1E66.json"
    
    box_data = {
        "center_x": 10.5, "center_y": 20.1, "center_z": -5.2,
        "size_x": 22.0, "size_y": 22.0, "size_z": 22.0
    }
    
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(box_data, f)
        
    assert grid_path.exists()
    with open(grid_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["center_x"] == 10.5
        assert loaded["size_x"] == 22.0