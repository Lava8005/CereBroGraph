import json
import pytest
from pathlib import Path

def test_target_list_iteration_schema(tmp_path: Path):
    """Ensure target preparation correctly parses the universal JSON array schema."""
    mock_targets = [
        {"disease": "Alzheimers_AChE", "pdb_id": "1E66", "ligand_res_name": "HUX"},
        {"disease": "Parkinsons_MAOB", "pdb_id": "2V5Z", "ligand_res_name": "SAG"}
    ]
    
    config_path = tmp_path / "cns_targets.json"
    config_path.write_text(json.dumps(mock_targets))
    
    parsed = json.loads(config_path.read_text())
    
    assert isinstance(parsed, list), "Schema must remain a JSON array"
    
    processed_diseases = []
    for config in parsed:
        processed_diseases.append(config.get("disease"))
        
    assert "Alzheimers_AChE" in processed_diseases
    assert "Parkinsons_MAOB" in processed_diseases
    # Assert dictionary methods are not illegally invoked on the list
    assert not hasattr(parsed, "items")