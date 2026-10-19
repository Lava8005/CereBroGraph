"""TDD Suite for Resumable, Optimized AutoDock Vina Execution."""

import csv
import threading
from pathlib import Path
from unittest.mock import patch

from physics_engine.dock_vina import process_molecule, get_completed_candidates

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
    """Verify 3D embedding runs exactly ONCE per molecule for 3 targets."""
    mock_prepare.return_value = "REMARK MOCK PDBQT DATA"
    mock_dock.side_effect = [
        ("Alzheimers_AChE", -9.0),
        ("Parkinsons_MAOB", -8.5),
        ("Schizophrenia_5HT2A", -8.0)
    ]
    
    row = {"candidate_code": "MOL-TEST", "smiles": "CCO"}
    targets = [{"disease": "Alzheimers_AChE"}, {"disease": "Parkinsons_MAOB"}, {"disease": "Schizophrenia_5HT2A"}]
    lock = threading.Lock()
    
    # THE FIX: output_csv=None replaces mock_csv=None
    result = process_molecule("vina", row, targets, tmp_path, output_csv=None, csv_lock=lock)
    
    mock_prepare.assert_called_once()
    assert mock_dock.call_count == 3
    assert '"Alzheimers_AChE": -9.0' in result["target_scores"]

def test_missing_target_fallback(tmp_path: Path):
    """Ensure un-embeddable molecules fail gracefully and return 0.0 for all targets."""
    with patch("physics_engine.dock_vina.prepare_ligand_pdbqt", return_value=None):
        row = {"candidate_code": "MOL-FAIL", "smiles": "INVALID"}
        targets = [{"disease": "T1"}, {"disease": "T2"}]
        lock = threading.Lock()
        
        result = process_molecule("vina", row, targets, tmp_path, output_csv=None, csv_lock=lock)
        assert '"T1": 0.0' in result["target_scores"]
        assert '"T2": 0.0' in result["target_scores"]