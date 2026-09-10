import sys
import pytest
from pathlib import Path

def test_dynamic_sys_path_injection(tmp_path: Path):
    """Ensure scripts can dynamically locate the project root regardless of execution context."""
    # Simulate the script's location
    script_dir = tmp_path / "vector_engine"
    script_dir.mkdir()
    script_file = script_dir / "train_eval.py"
    
    # Calculate root from the script's perspective
    calculated_root = str(script_file.resolve().parent.parent)
    
    # Inject
    if calculated_root not in sys.path:
        sys.path.insert(0, calculated_root)
        
    assert calculated_root in sys.path
    assert sys.path[0] == calculated_root  # Priority injection