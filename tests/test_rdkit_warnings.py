import pytest
import io
import sys
from vector_engine.b3db_to_graph import smiles_to_graph

def test_featurizer_silent_execution(monkeypatch):
    """Ensure the modern API executes without dumping deprecation warnings to stderr."""
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', captured_stderr)
    
    # Process a complex molecule to trigger multiple valence checks
    graph = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O") # Aspirin
    
    assert graph is not None
    assert graph.x.shape[0] > 0
    
    # Assert that no deprecation warnings flooded the console
    stderr_output = captured_stderr.getvalue()
    assert "DEPRECATION WARNING" not in stderr_output