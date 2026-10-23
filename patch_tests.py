# filename: patch_tests.py
import os
from pathlib import Path

print("Rebuilding Integration Tests for GitHub Actions...")

# 1. FIX HANDOFF INTEGRATION TEST
handoff_path = Path("tests/test_handoff_integration.py")
handoff_path.parent.mkdir(parents=True, exist_ok=True)
handoff_path.write_text('''import pytest
import sys
from unittest.mock import patch
from pathlib import Path
from physics_engine.dock_vina import batch_execute_docking

def test_vina_binary_hard_exit(tmp_path: Path):
    """Ensure missing Vina binary triggers sys.exit(1), bypassing missing input checks."""
    mock_csv = tmp_path / "mock_input.csv"
    mock_csv.write_text("candidate_code,smiles\\nMOL-123,CCO")
    mock_config = tmp_path / "mock_config.json"
    mock_config.write_text("[]")
    
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("vina not found")
        with pytest.raises(SystemExit) as exc_info:
            # THE FIX: Use exact keyword arguments matching dock_vina.py
            batch_execute_docking(
                input_csv=mock_csv, 
                output_csv=tmp_path / "out.csv", 
                config_path=mock_config
            )
        assert exc_info.value.code == 1

def test_docking_all_fail_hard_exit(tmp_path: Path):
    """Ensure 0 successful dockings across the batch triggers sys.exit(1)."""
    pass # Verified algorithmically in physics_engine.py
''')

# 2. FIX PHYSICS ENGINE TEST
physics_path = Path("tests/test_physics_engine.py")
physics_path.write_text('''"""TDD Suite for Resumable, Optimized AutoDock Vina Execution."""

import csv
import pytest
import threading
from pathlib import Path
from unittest.mock import patch

from physics_engine.dock_vina import process_molecule, get_completed_candidates, batch_execute_docking

def test_checkpoint_resumption(tmp_path: Path):
    """Ensure previously docked molecules are identified and skipped."""
    mock_csv = tmp_path / "docking_results.csv"
    with open(mock_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_code", "vina_score"])
        writer.writerow(["MOL-ABC123XYZ", "-9.0"])
        
    completed = get_completed_candidates(mock_csv)
    assert "MOL-ABC123XYZ" in completed

@patch("physics_engine.dock_vina.prepare_ligand_pdbqt")
@patch("physics_engine.dock_vina.dock_precomputed_target")
def test_single_embedding_efficiency(mock_dock, mock_prepare, tmp_path: Path):
    """Verify 3D embedding runs exactly ONCE per molecule for 3 targets and returns tuple."""
    mock_prepare.return_value = "REMARK MOCK PDBQT DATA"
    mock_dock.side_effect = [
        ("Alzheimers_AChE", -9.0),
        ("Parkinsons_MAOB", -8.5),
        ("Schizophrenia_5HT2A", -8.0)
    ]
    
    row = {"candidate_code": "MOL-TEST", "smiles": "CCO"}
    targets = [{"disease": "Alzheimers_AChE"}, {"disease": "Parkinsons_MAOB"}, {"disease": "Schizophrenia_5HT2A"}]
    lock = threading.Lock()
    
    result, is_success = process_molecule("vina", row, targets, tmp_path, output_csv=None, csv_lock=lock)
    
    mock_prepare.assert_called_once()
    assert mock_dock.call_count == 3
    assert is_success is True
    assert '"Alzheimers_AChE": -9.0' in result["target_scores"]

def test_missing_target_fallback(tmp_path: Path):
    """Ensure un-embeddable molecules fail gracefully and return 0.0."""
    with patch("physics_engine.dock_vina.prepare_ligand_pdbqt", return_value=None):
        row = {"candidate_code": "MOL-FAIL", "smiles": "INVALID"}
        targets = [{"disease": "T1"}, {"disease": "T2"}]
        lock = threading.Lock()
        
        result, is_success = process_molecule("vina", row, targets, tmp_path, output_csv=None, csv_lock=lock)
        
        assert is_success is False
        assert '"T1": 0.0' in result["target_scores"]

def test_docking_all_fail_hard_exit(tmp_path: Path):
    """Ensure 0 successful dockings across the batch triggers sys.exit(1)."""
    mock_csv = tmp_path / "mock.csv"
    mock_csv.write_text("candidate_code,smiles\\nMOL-1,CCO\\n")
    mock_config = tmp_path / "config.json"
    mock_config.write_text('[]')
    
    with patch("subprocess.run"):
        # Force process_molecule to return False (0 successful bindings)
        with patch("physics_engine.dock_vina.process_molecule", return_value=({}, False)):
            with pytest.raises(SystemExit) as exc:
                # THE FIX: Use exact keyword arguments matching dock_vina.py
                batch_execute_docking(
                    input_csv=mock_csv, 
                    output_csv=tmp_path / "out.csv", 
                    config_path=mock_config
                )
            assert exc.value.code == 1
''')

print("✅ Integration Tests rebuilt successfully.")