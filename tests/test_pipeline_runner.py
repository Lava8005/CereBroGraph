"""TDD Suite for Master Orchestrator State Management."""

import pytest
from pathlib import Path
from run_pipeline import is_stale

def test_cache_invalidation_logic():
    """Ensure multi-input cache invalidation accurately detects stale artifacts."""
    class MockFile:
        def __init__(self, exists=True, mtime=100):
            self._exists = exists
            self.mtime = mtime
        def exists(self): return self._exists
        def stat(self): return type('obj', (object,), {'st_mtime': self.mtime})
        
    old_input1 = MockFile(mtime=50)
    new_input2 = MockFile(mtime=200) # Simulating a newly trained GNN weights file
    output = MockFile(mtime=100)
    
    # Even if one input is old, if any input is newer than the output, it MUST be stale
    assert is_stale([old_input1, new_input2], output) is True
    assert is_stale([old_input1], output) is False

def test_missing_marker_invalidation():
    """Ensure missing marker files force a stage re-run."""
    class MockFile:
        def __init__(self, exists=True):
            self._exists = exists
        def exists(self): return self._exists
        
    missing_marker = MockFile(exists=False)
    input_file = MockFile(exists=True)
    
    assert is_stale([input_file], missing_marker) is True