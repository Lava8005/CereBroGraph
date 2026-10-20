"""TDD Suite for Data Leakage Prevention and Subprocess State Management."""

import pytest
import sys
from unittest.mock import patch
from physics_engine.dock_vina import batch_execute_docking
from vector_engine.b3db_loader import FDA_CONTROL_NAMES

def test_vina_binary_hard_exit():
    """Ensure a missing Vina binary triggers a hard sys.exit(1) to prevent false orchestrator markers."""
    with patch("subprocess.run") as mock_run:
        # Simulate Vina missing from system PATH
        mock_run.side_effect = FileNotFoundError("vina not found")
        
        with pytest.raises(SystemExit) as exc_info:
            batch_execute_docking(config_path="mock_config.json")
            
        assert exc_info.value.code == 1

def test_dual_layer_quarantine_strings():
    """Ensure substring matching catches homologous variations of controls."""
    homolog_name = "Loperamide-derivative (4-carbon)"
    
    # Topological mapping might fail if the SMILES differs, but string matching MUST catch it
    is_quarantined = any(n in homolog_name.upper() for n in FDA_CONTROL_NAMES)
    assert is_quarantined is True