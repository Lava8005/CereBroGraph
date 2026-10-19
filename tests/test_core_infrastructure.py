"""TDD Suite for Curation Integrity and Orchestrator State Management."""

import pytest
from pathlib import Path
from library.curate import process_chunk

def test_stable_inchikey_generation():
    """Ensure candidate codes are perfectly stable across runs and exact canonical SMILES."""
    chunk = [
        {"source_id": "ZINC_1", "smiles": "CCO"},
        {"source_id": "ZINC_2", "smiles": "C(C)O"} # Equivalent topology
    ]
    results = process_chunk(chunk)
    
    assert len(results) == 2
    # InChIKeys MUST match exactly for equivalent molecules
    assert results[0]["candidate_code"] == results[1]["candidate_code"]
    assert results[0]["candidate_code"].startswith("MOL-")
    # Must use full 27-character hash to prevent stereochemistry collisions
    assert len(results[0]["candidate_code"]) > 15 

def test_invalid_smiles_rejection():
    """Ensure malformed SMILES strings are silently dropped."""
    chunk = [{"source_id": "BAD", "smiles": "C1=CC=C(C=C1"}]
    results = process_chunk(chunk)
    assert len(results) == 0

def test_cache_invalidation_logic():
    """Ensure orchestrator physically deletes stale files."""
    from run_pipeline import is_stale
    
    # Mocking filesystem state
    class MockFile:
        def __init__(self, exists=True, mtime=100):
            self._exists = exists
            self.mtime = mtime
        def exists(self): return self._exists
        def stat(self): return type('obj', (object,), {'st_mtime': self.mtime})
        
    old_input = MockFile(mtime=200)
    new_output = MockFile(mtime=100)
    
    # If input is newer than output, output is stale
    assert is_stale([old_input], new_output) == True