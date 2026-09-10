import pytest
import pandas as pd
import numpy as np
from ranking.pareto_front import compute_pareto_front

def test_fast_pareto_front():
    """Ensure the optimized sweep strictly matches brute-force Pareto logic."""
    # Construct a dataframe with known dominating and dominated points
    df = pd.DataFrame([
        {"candidate_code": "A", "vina_score": -12.0, "bbb_probability": 0.9, "qed": 0.8}, # Dominates C
        {"candidate_code": "B", "vina_score": -11.0, "bbb_probability": 0.95, "qed": 0.7}, # Pareto
        {"candidate_code": "C", "vina_score": -10.0, "bbb_probability": 0.8, "qed": 0.7}, # Dominated by A
    ])
    
    result_df = compute_pareto_front(df)
    
    assert result_df.loc[result_df["candidate_code"] == "A", "on_pareto_front"].iloc[0] == True
    assert result_df.loc[result_df["candidate_code"] == "B", "on_pareto_front"].iloc[0] == True
    assert result_df.loc[result_df["candidate_code"] == "C", "on_pareto_front"].iloc[0] == False