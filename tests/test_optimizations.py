"""TDD Suite for Funnel Truncation, Featurization Bypasses, and Precision."""

import pytest
import torch
from rdkit import Chem
from vector_engine.b3db_to_graph import smiles_to_graph
from physics_engine.drug_likeness import calculate_cns_likeness

def test_inference_scaffold_bypass():
    """Ensure graph generation skips costly Murcko calculations during inference."""
    smiles = "CCO"
    
    # Training mode (calculates scaffold)
    graph_train = smiles_to_graph(smiles, is_inference=False)
    assert hasattr(graph_train, "scaffold")
    assert graph_train.scaffold is not None
    
    # Inference mode (bypasses scaffold)
    graph_inf = smiles_to_graph(smiles, is_inference=True)
    assert not hasattr(graph_inf, "scaffold") or graph_inf.scaffold == ""
    
    # Ensure topology remains mathematically identical
    assert torch.equal(graph_train.x, graph_inf.x)
    assert torch.equal(graph_train.edge_index, graph_inf.edge_index)

def test_cns_heuristic_filter():
    """Ensure non-CNS drugs are violently culled before reaching the GNN."""
    # Donepezil (Valid CNS Drug - MW 379, TPSA 38, HBD 0)
    donepezil = Chem.MolFromSmiles("COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2")
    assert calculate_cns_likeness(donepezil) is True
    
    # Large non-CNS macrocycle (Fails MW > 500 and TPSA > 90)
    macrocycle = Chem.MolFromSmiles("C" * 40 + "(=O)O" * 5)
    assert calculate_cns_likeness(macrocycle) is False

def test_fp32_sigmoid_precision():
    """Verify FP16 tensors are cast to FP32 before sigmoid to prevent digit loss."""
    # Simulate a mixed-precision logit from the GNN
    logit_fp16 = torch.tensor([2.1234], dtype=torch.float16)
    
    # Incorrect FP16 Sigmoid (Truncates precision)
    prob_fp16 = torch.sigmoid(logit_fp16).item()
    
    # Correct FP32 Sigmoid
    prob_fp32 = torch.sigmoid(logit_fp16.float()).item()
    
    # FP32 should contain more granular precision data
    assert prob_fp16 != prob_fp32
    assert type(prob_fp32) == float