import json
import pytest
from pathlib import Path

def test_full_polypharmacology_matrix():
    """Ensure all three required CNS targets are registered with correct active site ligands."""
    config_path = Path("config/cns_targets.json")
    targets = json.loads(config_path.read_text(encoding="utf-8"))
    
    assert len(targets) == 3
    
    target_map = {t["disease"]: t for t in targets}
    
    assert target_map["Alzheimers_AChE"]["pdb_id"] == "1E66"
    assert target_map["Alzheimers_AChE"]["ligand_res_name"] == "HUX"
    
    assert target_map["Parkinsons_MAOB"]["pdb_id"] == "2V5Z"
    assert target_map["Parkinsons_MAOB"]["ligand_res_name"] == "SAG"
    
    assert target_map["Schizophrenia_5HT2A"]["pdb_id"] == "6A93"
    assert target_map["Schizophrenia_5HT2A"]["ligand_res_name"] == "RIS"