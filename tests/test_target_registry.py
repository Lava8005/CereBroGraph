import json
import pytest
from pathlib import Path

def test_full_polypharmacology_matrix():
    """Ensure all three required CNS targets are registered with the expected PDB entries."""
    config_path = Path("config/cns_targets.json")
    targets = json.loads(config_path.read_text(encoding="utf-8"))
    
    assert len(targets) == 3
    
    target_map = {t["disease"]: t for t in targets}
    
    assert target_map["Alzheimers_AChE"]["pdb_id"] == "4EY7"
    assert target_map["Parkinsons_MAOB"]["pdb_id"] == "2V5Z"
    assert target_map["Schizophrenia_5HT2A"]["pdb_id"] == "6A93"

    for t in targets:
        size = t["active_site"]
        assert all(size[k] > 0 for k in ("size_x", "size_y", "size_z"))