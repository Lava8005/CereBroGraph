import pandas as pd
from ranking.pipeline_ranker import extract_weighted_affinity
from ranking.pareto_front import compute_pareto_front

def test_vina_score_unweighted_fallback():
    scores = {"TargetA": -8.0, "TargetB": -6.0}
    assert extract_weighted_affinity(scores, {}) == -7.0

def test_ranker_truncation():
    df = pd.DataFrame([
        {"source_id": "MolA", "vina_score": -9.0, "bbb_probability": 0.99, "qed": 0.8},
        {"source_id": "MolB", "vina_score": -6.0, "bbb_probability": 0.40, "qed": 0.5}
    ])
    result = compute_pareto_front(df)
    assert result[result["source_id"] == "MolA"]["on_pareto_front"].iloc[0] == True
