import pytest
import io
import sys
import torch
from pathlib import Path
from vector_engine.b3db_to_graph import smiles_to_graph, get_b3db_dataloaders

def test_featurizer_topology_and_silence(monkeypatch):
    """Ensure RDKit correctly translates topology using modern APIs without console flooding."""
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', captured_stderr)
    
    graph = smiles_to_graph("CCO")
    assert graph is not None
    assert graph.x.shape == (3, 5) 
    assert graph.edge_index.shape == (2, 4)
    assert "DEPRECATION WARNING" not in captured_stderr.getvalue()

def test_tsv_stratified_loader(tmp_path: Path):
    """Ensure the dataloader correctly parses TSV schemas and drops malformed data."""
    mock_tsv = tmp_path / "B3DB_classification.tsv"
    mock_tsv.write_text("SMILES\tBBB+/BBB-\nCCO\tBBB+\nCCN\tBBB-\nINVALID\tBBB+\n")
    
    train_loader, val_loader = get_b3db_dataloaders(csv_path=mock_tsv, batch_size=2, val_split=0.0)
    
    assert len(train_loader.dataset) == 2  # The 'INVALID' string is safely dropped
    for batch in train_loader:
        assert isinstance(batch.x, torch.Tensor)
        assert set(batch.y.view(-1).tolist()).issubset({0.0, 1.0})
        break