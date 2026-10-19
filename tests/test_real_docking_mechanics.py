"""TDD Suite for Real 3D Conformer Generation and AutoDock Vina Execution."""

import pytest
from pathlib import Path
from unittest.mock import patch
from physics_engine.dock_vina import prepare_ligand_pdbqt, dock_single_target

def test_rdkit_meeko_pdbqt_generation():
    """Ensure RDKit and Meeko successfully fold SMILES into 3D PDBQT strings."""
    # Ethanol (Simple molecule)
    smiles = "CCO"
    pdbqt = prepare_ligand_pdbqt(smiles)
    
    assert pdbqt is not None
    assert "ATOM" in pdbqt
    assert "ROOT" in pdbqt
    assert "TORSDOF" in pdbqt

@patch("subprocess.run")
def test_dock_single_target_execution(mock_run, tmp_path: Path):
    """Ensure Vina subprocess is invoked with precise 3D grid bounds and temporary files."""
    # Mock Vina's stdout table format
    mock_run.return_value.stdout = "   1       -10.500      0.000      0.000"
    
    vina_exe = Path("vina")
    grid = {"center_x": 1.0, "center_y": 2.0, "center_z": 3.0, "size_x": 10, "size_y": 10, "size_z": 10}
    
    grid_file = tmp_path / "grid.json"
    import json
    grid_file.write_text(json.dumps(grid))
    
    rec_file = tmp_path / "rec.pdbqt"
    rec_file.write_text("REMARK MOCK RECEPTOR")
    
    target_config = {"disease": "Test_Disease", "pdbqt": str(rec_file), "grid": str(grid_file)}
    
    disease, score = dock_single_target(vina_exe, "CCO", "OV-01", target_config, tmp_path)
    
    assert disease == "Test_Disease"
    assert score == -10.5
    
    # Verify exact CLI args sent to OS
    cmd = mock_run.call_args[0][0]
    assert "--center_x" in cmd
    assert cmd[cmd.index("--center_x") + 1] == "1.0"
    assert "--exhaustiveness" in cmd
    assert "--ligand" in cmd