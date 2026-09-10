import pytest
import json
from pathlib import Path
from unittest.mock import patch
from physics_engine.dock_vina import batch_dock_library

@patch("physics_engine.dock_vina.execute_single_dock")
@patch("physics_engine.dock_vina.ensure_vina_binary")
def test_polypharmacology_docking(mock_ensure_binary, mock_single_dock, tmp_path: Path):
    """Ensure multi-target binding affinities are aggregated correctly per candidate."""
    mock_ensure_binary.return_value = tmp_path / "mock_vina"
    
    def mock_dock_side_effect(vina_exe, receptor_pdbqt, *args, **kwargs):
        if "AChE" in str(receptor_pdbqt): return -10.0
        if "MAOB" in str(receptor_pdbqt): return -8.5
        return -5.0
        
    mock_single_dock.side_effect = mock_dock_side_effect
    
    filtered_csv = tmp_path / "filtered_library.csv"
    filtered_csv.write_text("candidate_code,smiles,qed,bbb_probability\nOV-000001,CCO,0.8,0.9\n")
    
    # Isolate multiple grid files via absolute path mapping
    grid_path_1 = tmp_path / "g1.json"
    grid_path_2 = tmp_path / "g2.json"
    grid_path_1.write_text('{"center_x":0,"center_y":0,"center_z":0,"size_x":10,"size_y":10,"size_z":10}')
    grid_path_2.write_text('{"center_x":0,"center_y":0,"center_z":0,"size_x":10,"size_y":10,"size_z":10}')
    
    target_idx = tmp_path / "target_index.json"
    target_data = [
        {"disease": "AChE", "pdbqt": "AChE_rec.pdbqt", "grid": str(grid_path_1)},
        {"disease": "MAOB", "pdbqt": "MAOB_rec.pdbqt", "grid": str(grid_path_2)}
    ]
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
    assert mock_single_dock.call_count == 2
    
    results_text = out_csv.read_text()
    assert "AChE" in results_text
    assert "MAOB" in results_text
    assert "-10.0" in results_text
    assert "-8.5" in results_text