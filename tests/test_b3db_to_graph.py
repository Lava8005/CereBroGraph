"""TDD Suite for Optimized Chemical Graph Featurization."""

import pytest
import torch
from vector_engine.b3db_to_graph import smiles_to_graph, get_node_dim, get_edge_dim

def test_inference_scaffold_bypass():
    """Ensure Murcko scaffold is bypassed during inference for maximum throughput."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
    
    graph_train = smiles_to_graph(smiles, is_inference=False)
    assert hasattr(graph_train, "scaffold")
    assert graph_train.scaffold != ""
    
    graph_inf = smiles_to_graph(smiles, is_inference=True)
    assert not hasattr(graph_inf, "scaffold") or graph_inf.scaffold == ""

def test_rich_featurization_dimensions():
    """Verify nodes, edges, and global descriptors generate the correct tensor shapes."""
    smiles = "c1ccccc1" # Benzene (6 atoms, 6 bonds)
    graph = smiles_to_graph(smiles, is_inference=True)
    
    # Verify Node Tensor: 6 atoms x N features
    assert graph.x.shape == (6, get_node_dim())
    
    # Verify Edge Index & Attributes (Bidirectional = 12 directed edges)
    assert graph.edge_index.shape == (2, 12)
    assert graph.edge_attr.shape == (12, get_edge_dim())
    
    # Verify Global Descriptors (TPSA, MW)
    assert hasattr(graph, "global_features")
    assert graph.global_features.shape == (1, 2)
    assert not torch.isnan(graph.global_features).any()

def test_invalid_smiles_handling():
    """Ensure malformed SMILES strings return None without crashing."""
    assert smiles_to_graph("INVALID_SMILES") is None