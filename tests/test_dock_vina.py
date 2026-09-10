import pytest
from pathlib import Path
from unittest.mock import patch
from physics_engine.dock_vina import execute_single_dock

@patch("subprocess.run")
def test_execute_single_dock(mock_run, tmp_path: Path):
    """Ensure single dock passes correct CPU and thread flags."""
    mock_run.return_value.stdout = "   1       -8.452      0.000      0.000"
    
    score = execute_single_dock(
        vina_exe=Path("vina"),
        receptor_pdbqt="rec.pdbqt",
        ligand_pdbqt=Path("lig.pdbqt"),
        grid_params={"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 10, "size_y": 10, "size_z": 10}
    )
    
    assert score == -8.452
    
    called_args = mock_run.call_args[0][0]
    assert "--cpu" in called_args
    assert called_args[called_args.index("--cpu") + 1] == "1"