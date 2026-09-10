import json
import pytest
from pathlib import Path
from unittest.mock import patch
from target.prepare_target import batch_prepare_cns_targets

@patch("target.prepare_target.prepare_single_target")
def test_batch_prepare_cns_targets(mock_prepare, tmp_path: Path):
    """Ensure the target preparator correctly parses the CerebroGraph registry."""
    # Setup mock registry
    registry_path = tmp_path / "cns_targets.json"
    registry_path.write_text(json.dumps({
        "Alzheimers_AChE": {"pdb_id": "1E66", "ligand_res_name": "HUX"},
        "Parkinsons_MAOB": {"pdb_id": "2V5Z", "ligand_res_name": "PIZ"}
    }))
    
    # Mock the return values for the single target execution
    mock_prepare.return_value = (Path("dummy.pdbqt"), Path("dummy_grid.json"))
    
    # Execute batch preparation
    results = batch_prepare_cns_targets(registry_path=registry_path, work_dir=tmp_path)
    
    assert len(results) == 2
    assert results[0]["disease"] == "Alzheimers_AChE"
    assert mock_prepare.call_count == 2
    
    # Verify the master index was written for downstream consumption
    assert (tmp_path / "target_index.json").exists()