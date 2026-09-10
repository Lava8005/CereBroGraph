import pytest
import sys
from pathlib import Path
from physics_engine.dock_vina import dock_single_target

def test_null_safe_path_casting():
    """Ensure missing registry keys are caught gracefully without crashing the ThreadPool."""
    # Simulating the static config that lacks 'pdbqt' and 'grid' keys
    bad_config = {"disease": "Alzheimers_AChE"} 
    
    # Execute - should return 0.0 safely instead of throwing TypeError
    disease, score = dock_single_target(Path("vina"), "CCO", "OV-01", bad_config, Path("work"))
    
    assert disease == "Alzheimers_AChE"
    assert score == 0.0

def test_ranker_sys_path_resolution(tmp_path: Path):
    """Ensure the ranker dynamically anchors itself to the project root."""
    ranker_file = tmp_path / "ranking" / "pipeline_ranker.py"
    calculated_root = str(ranker_file.resolve().parent.parent)
    
    if calculated_root not in sys.path:
        sys.path.insert(0, calculated_root)
        
    assert calculated_root in sys.path