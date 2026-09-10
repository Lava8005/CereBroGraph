import pytest
import sys
from pathlib import Path
from run_pipeline import is_stale, run_stage, PROJECT_ROOT

def test_freshness_check(tmp_path: Path):
    """Ensure the state machine accurately detects when to skip redundant compute."""
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"
    
    in_file.touch()
    assert is_stale([in_file], out_file) is True # Output missing, must run
    
    out_file.touch()
    in_file.write_text("new data")
    assert is_stale([in_file], out_file) is True # Input modified, must run
    
def test_path_resolution():
    """Ensure the orchestrator calculates strict absolute paths for subprocesses."""
    assert PROJECT_ROOT.is_absolute()
    assert (PROJECT_ROOT / "library" / "curate.py").is_absolute()