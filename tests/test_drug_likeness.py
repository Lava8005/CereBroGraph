import pytest
from rdkit import Chem
from physics_engine.drug_likeness import calculate_lipinski

def test_lipinski_rule_of_five():
    """Ensure heuristic correctly flags violations."""
    aspirin = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert calculate_lipinski(aspirin) == 0