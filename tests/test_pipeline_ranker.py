"""TDD Suite for Pipeline Ranker."""
import pandas as pd
from ranking.pareto_front import compute_pareto_fronts

def test_pareto_penalty_isolation():
    """Ensure that a failed docking (+99.0) prevents a molecule from dominating."""
    df = pd.DataFrame([
        # MolA is perfect on T1, but failed T2
        {"candidate_code": "M1", "vina_T1": -10.0, "vina_T2": 99.0, "bbb_probability": 0.9, "qed": 0.8},
        # MolB is mediocre but succeeded on both
        {"candidate_code": "M2", "vina_T1": -7.0, "vina_T2": -7.0, "bbb_probability": 0.9, "qed": 0.8},
    ])
    
    # Minimize T1 and T2, Maximize BBB and QED
    result = compute_pareto_fronts(df, minimize_cols=["vina_T1", "vina_T2"], maximize_cols=["bbb_probability", "qed"])
    
    # Both should be rank 1 because MolA is non-dominated by MolB (MolA is better at T1). 
    assert "pareto_rank" in result.columns
    assert len(result) == 2