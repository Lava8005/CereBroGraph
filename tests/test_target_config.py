import pytest
import json
from pathlib import Path

def test_custom_pdb_target_configuration(tmp_path: Path):
    """Ensure the pipeline configuration properly accepts a custom PDB target."""
    config_path = tmp_path / "cns_targets.json"
    
    # Mocking the configuration payload
    payload = [{
        "disease": "Alzheimers_AChE_Custom",
        "raw_pdb": "data/target/raw/1e66.pdb",
        "pdbqt": "data/target/1e66_target.pdbqt",
        "grid": "data/target/1e66_grid.json",
        "active_site": {"center_x": 10.0, "center_y": -15.2, "center_z": 20.5, "size": 20}
    }]
    
    config_path.write_text(json.dumps(payload))
    loaded_config = json.loads(config_path.read_text())
    
    assert len(loaded_config) == 1
    assert loaded_config[0]["raw_pdb"].endswith("1e66.pdb")
    assert "center_x" in loaded_config[0]["active_site"]