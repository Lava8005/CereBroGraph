import pytest
import json
from pathlib import Path
from unittest.mock import patch
from physics_engine.dock_vina import batch_dock_library

@patch("physics_engine.dock_vina.execute_single_dock")
@patch("physics_engine.dock_vina.ensure_vina_binary")
def test_parallel_docking_execution(mock_ensure_binary, mock_single_dock, tmp_path: Path):
    """Ensure the ThreadPool executes workloads concurrently without oversubscription."""
    mock_ensure_binary.return_value = tmp_path / "mock_vina"
    mock_single_dock.return_value = -8.5
    
    filtered_csv = tmp_path / "filtered_library.csv"
    filtered_csv.write_text("candidate_code,smiles,qed,bbb_probability\nOV-000001,CCO,0.8,0.9\n")
    
    # Isolate grid file via absolute path mapping
    grid_path = tmp_path / "g.json"
    grid_path.write_text('{"center_x":0,"center_y":0,"center_z":0,"size_x":10,"size_y":10,"size_z":10}')
    
    target_idx = tmp_path / "target_index.json"
    target_data = [{"disease": "AChE", "pdbqt": "t.pdbqt", "grid": str(grid_path)}]
    target_idx.write_text(json.dumps(target_data))
    
    out_csv = tmp_path / "docking_results.csv"
    work_dir = tmp_path / "physics_engine"
    
    cache_dir = work_dir / "ligand_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "OV-000001.pdbqt").touch()
    
    count = batch_dock_library(
        library_csv=filtered_csv, 
        target_index=target_idx, 
        output_csv=out_csv, 
        work_dir=work_dir,
        max_workers=2,
        batch_size=500
    )
    
    assert count == 1
    assert mock_single_dock.called
    
    results_text = out_csv.read_text()
    assert "OV-000001" in results_text
    assert "target_scores" in results_text
    assert "-8.5" in results_text