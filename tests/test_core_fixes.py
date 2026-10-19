import pandas as pd
import numpy as np
from rdkit import Chem
from ranking.pareto_front import compute_pareto_front
from ranking.pipeline_ranker import extract_weighted_affinity

def test_lexicographical_pareto_ties():
    """Ensure ties in Vina score are correctly dominated by better BBB/QED."""
    df = pd.DataFrame([
        # Mol A and B tie in Vina. A is strictly better in BBB/QED.
        # If sorted wrong, B might enter the front first and stay there.
        {"source_id": "Mol_B_Dominated", "vina_score": -8.0, "bbb_probability": 0.5, "qed": 0.5},
        {"source_id": "Mol_A_Dominator", "vina_score": -8.0, "bbb_probability": 0.9, "qed": 0.9},
    ])
    
    # Run the updated pareto sweep
    result = compute_pareto_front(df)
    
    front_ids = result[result["on_pareto_front"] == True]["source_id"].tolist()
    assert "Mol_A_Dominator" in front_ids
    assert "Mol_B_Dominated" not in front_ids

def test_missing_target_penalty():
    """Ensure failed targets explicitly penalize the weighted average."""
    # Only 1 out of 3 targets succeeded. 
    scores = {"Alzheimers_AChE": -9.0} 
    weights = {"Alzheimers_AChE": 1.0, "Parkinsons_MAOB": 1.0, "Schizophrenia_5HT2A": 1.0}
    
    # Old logic would do: -9.0 / 1 = -9.0. 
    # New logic MUST do: (-9.0 + 0 + 0) / 3 = -3.0.
    weighted_score = extract_weighted_affinity(scores, weights)
    assert weighted_score == -3.0

def test_stable_inchikey_generation():
    """Ensure candidate codes are stable across runs."""
    mol = Chem.MolFromSmiles("CCO")
    key1 = Chem.inchi.MolToInchiKey(mol)
    key2 = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles("C(C)O")) # Equivalent SMILES
    assert key1 == key2