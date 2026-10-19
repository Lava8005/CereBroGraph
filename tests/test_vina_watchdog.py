"""TDD Suite for Physics Engine Thread Scaling and Subprocess Timeouts."""

import pytest
import os
import subprocess
from unittest.mock import patch
from pathlib import Path
from physics_engine.dock_vina import dock_single_target

@patch("subprocess.run")
def test_subprocess_timeout_handling(mock_run, tmp_path: Path):
    """Ensure that a hanging AutoDock Vina subprocess is caught and handled safely."""
    # Simulate a subprocess timeout
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="vina", timeout=45)
    
    target_config = {
        "disease": "Timeout_Disease",
        "pdbqt": str(tmp_path / "mock.pdbqt"),
        "grid": str(tmp_path / "grid.json")
    }
    
    # Touch mock files to bypass validation
    (tmp_path / "mock.pdbqt").touch()
    (tmp_path / "grid.json").write_text("{}")
    
    disease, score = dock_single_target(Path("vina"), "CCO", "OV-01", target_config, tmp_path)
    
    assert disease == "Timeout_Disease"
    assert score == 0.0  # The pipeline must safely return 0.0 and not crash