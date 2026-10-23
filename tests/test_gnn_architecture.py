"""TDD Suite for GINE Architecture and Tensor Routing."""

import pytest
import torch
from vector_engine.gnn_model import BBBGraphModel

def test_gine_forward_pass():
    """Ensure the new GINE architecture digests nodes, edges, and globals correctly."""
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
    model.eval()
    
    # Mock Batch: 2 graphs, 5 total nodes, 4 total directed edges
    x = torch.randn(5, 27)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 7)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    global_features = torch.randn(2, 2)
    
    with torch.no_grad():
        logits = model(x, edge_index, batch, edge_attr, global_features)
        
    # Output must be a logit per graph (2, 1)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()

def test_missing_globals_fallback():
    """Ensure the model doesn't crash if global features are temporarily absent."""
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=0, hidden_dim=64)
    model.eval()  # BatchNorm needs eval mode for a single graph
    x = torch.randn(3, 27)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 7)
    batch = torch.tensor([0, 0, 0], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x, edge_index, batch, edge_attr=edge_attr, global_features=None)
    assert logits.shape == (1, 1)