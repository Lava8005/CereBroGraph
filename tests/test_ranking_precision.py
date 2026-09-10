import pytest
import pandas as pd
from ranking.multi_objective import compute_composite_scores

def test_composite_score_rounding():
    """Ensure floating point artifacts are truncated for UI rendering and DB storage."""
    df = pd.DataFrame([
        {"candidate_code": "A", "vina_score": -10.0, "bbb_probability": 0.8184, "qed": 0.7475},
        {"candidate_code": "B", "vina_score": -9.0, "bbb_probability": 0.6030, "qed": 0.8586}
    ])
    
    df_scored = compute_composite_scores(df)
    score_a = df_scored.loc[df_scored["candidate_code"] == "A", "composite_score"].iloc[0]
    
    # Must be exactly 4 decimal places, avoiding 0.39856499999999995
    assert len(str(score_a).split('.')[1]) <= 4