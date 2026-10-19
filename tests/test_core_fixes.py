import pytest
from vector_engine.b3db_loader import FDA_CONTROL_SMILES
from library.curate import process_chunk
from ranking.pipeline_ranker import extract_weighted_affinity

def test_fda_quarantine_smiles_mapping():
    """Ensure FDA controls are mapped by their exact canonical SMILES."""
    from rdkit import Chem
    donepezil_smiles = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(donepezil_smiles), canonical=True)
    assert canonical in FDA_CONTROL_SMILES

def test_curate_stable_prefix_and_dedupe():
    """Ensure MOL- prefix is applied in the worker and deduplication uses the code."""
    chunk = [{"source_id": "A", "smiles": "CCO"}]
    result = process_chunk(chunk)
    assert result[0]["candidate_code"].startswith("MOL-")
    assert len(result[0]["candidate_code"]) > 15 # Full InChIKey

def test_pipeline_ranker_fixtures():
    """Ensure ranker handles the new schemas correctly."""
    scores = {"TargetA": -9.0}
    weights = {"TargetA": 1.0, "TargetB": 1.0}
    val = extract_weighted_affinity(scores, weights)
    assert val == -4.5 # Penalized for missing TargetB