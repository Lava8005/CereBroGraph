"""TDD Suite for Library Curation and Deterministic Key Generation."""

import pytest
from rdkit import Chem
from library.curate import process_chunk

def test_stable_inchikey_generation():
    """Ensure candidate codes are completely stable across runs and matching SMILES."""
    chunk = [
        {"source_id": "ZINC_1", "smiles": "CCO"},  # Ethanol (Standard)
        {"source_id": "ZINC_2", "smiles": "C(C)O"} # Ethanol (Non-standard SMILES)
    ]
    
    results = process_chunk(chunk)
    
    assert len(results) == 2
    # Both variations of Ethanol MUST yield the exact same deterministic candidate_code
    assert results[0]["candidate_code"] == results[1]["candidate_code"]
    assert results[0]["candidate_code"].startswith("MOL-")
    assert results[0]["smiles"] == "CCO" # Both canonicalized

def test_invalid_smiles_rejection():
    """Ensure malformed SMILES strings are silently dropped without crashing the worker."""
    chunk = [
        {"source_id": "GOOD_MOL", "smiles": "c1ccccc1"},
        {"source_id": "BAD_MOL", "smiles": "C1=CC=C(C=C1"} # Unclosed ring
    ]
    
    results = process_chunk(chunk)
    
    assert len(results) == 1
    assert results[0]["source_id"] == "GOOD_MOL"

def test_missing_data_handling():
    """Ensure missing fields fallback gracefully."""
    chunk = [{"smiles": "C"}] # Missing source_id
    results = process_chunk(chunk)
    assert results[0]["source_id"] == "UNKNOWN"