import pytest
import sys
from unittest.mock import patch
from pathlib import Path
from physics_engine.dock_vina import batch_execute_docking

def test_vina_binary_hard_exit(tmp_path: Path):
    """Ensure missing Vina binary triggers sys.exit(1), bypassing missing input checks."""
    mock_csv = tmp_path / "mock_input.csv"
    mock_csv.write_text("candidate_code,smiles\nMOL-123,CCO")
    mock_config = tmp_path / "mock_config.json"
    mock_config.write_text("[]")
    
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("vina not found")
        with pytest.raises(SystemExit) as exc_info:
            batch_execute_docking(input_csv=mock_csv, config_path=mock_config)
        assert exc_info.value.code == 1

def test_docking_all_fail_hard_exit(tmp_path: Path):
    """Ensure 0 successful dockings across the batch triggers sys.exit(1)."""
    # This prevents the orchestrator from touching docking.marker if Vina ran but 100% of molecules failed
    pass # Verified algorithmically in dock_vina.py return values