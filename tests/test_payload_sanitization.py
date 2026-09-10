import pytest
from ranking.pipeline_ranker import parse_target_scores, extract_best_vina

def test_jsonb_deserialization():
    """Ensure stringified CSV dictionaries are parsed into valid Python dicts for Supabase JSONB."""
    raw_csv_val = '{"Alzheimers_AChE": -8.5, "Parkinsons_MAOB": -9.2}'
    parsed = parse_target_scores(raw_csv_val)
    
    assert isinstance(parsed, dict)
    assert parsed["Parkinsons_MAOB"] == -9.2

def test_vina_score_extraction():
    """Ensure the minimum binding affinity is correctly routed to the dedicated numeric column."""
    scores = {"AChE": -8.5, "MAOB": -10.1, "5HT2A": -7.2}
    best_score = extract_best_vina(scores)
    
    assert best_score == -10.1