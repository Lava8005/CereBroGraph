import pytest
import pandas as pd
from ranking.multi_objective import compute_composite_scores
from ranking.pareto_front import compute_pareto_front

def test_composite_score_calculation_dataframe():
    """Ensure multi-objective math correctly evaluates wrapped DataFrames."""
    candidates = [
        {"candidate_code": "OV-000001", "vina_score": -10.0, "bbb_probability": 0.9, "qed": 0.8},
        {"candidate_code": "OV-000002", "vina_score": -5.0,  "bbb_probability": 0.4, "qed": 0.5}
    ]
    
    df = pd.DataFrame(candidates)
    df_scored = compute_composite_scores(df)
    
    assert "composite_score" in df_scored.columns
    score_1 = df_scored.loc[df_scored["candidate_code"] == "OV-000001", "composite_score"].iloc[0]
    score_2 = df_scored.loc[df_scored["candidate_code"] == "OV-000002", "composite_score"].iloc[0]
    
    assert score_1 > score_2

def test_pareto_front_dataframe():
    """Ensure the sorted skyline sweep identifies the Pareto front correctly."""
    df_scored = pd.DataFrame([
        {"candidate_code": "A", "vina_score": -12.0, "bbb_probability": 0.9, "qed": 0.8}, # Dominates C
        {"candidate_code": "B", "vina_score": -11.0, "bbb_probability": 0.95, "qed": 0.7}, # Pareto
        {"candidate_code": "C", "vina_score": -10.0, "bbb_probability": 0.8, "qed": 0.7}, # Dominated
    ])
    
    df_pareto = compute_pareto_front(df_scored)
    
    # Use value equality (==) to safely handle numpy.bool_ evaluations
    assert df_pareto.loc[df_pareto["candidate_code"] == "A", "on_pareto_front"].iloc[0] == True
    assert df_pareto.loc[df_pareto["candidate_code"] == "B", "on_pareto_front"].iloc[0] == True
    assert df_pareto.loc[df_pareto["candidate_code"] == "C", "on_pareto_front"].iloc[0] == False