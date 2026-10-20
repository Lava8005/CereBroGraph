"""TDD Suite for Skyline Pareto Ranking and Queue Truncation."""

import pandas as pd
from ranking.pareto_front import compute_pareto_front

def test_ranker_truncation_logic():
    """Ensure ranker accurately flags the Pareto front using all mathematical columns."""
    # Injected bbb_probability and qed to satisfy the Pareto mathematical constraints
    df = pd.DataFrame([
        {"source_id": "MolA", "vina_score": -9.0, "bbb_probability": 0.99, "qed": 0.8},
        {"source_id": "MolB", "vina_score": -6.0, "bbb_probability": 0.40, "qed": 0.5},
        {"source_id": "MolC", "vina_score": -2.0, "bbb_probability": 0.10, "qed": 0.2}
    ])
    
    result = compute_pareto_front(df)
    
    # Verify the column was successfully attached
    assert "on_pareto_front" in result.columns
    
    # MolA strictly dominates MolB and MolC
    assert result[result["source_id"] == "MolA"]["on_pareto_front"].iloc[0] is True
    # MolC is completely dominated and should not be on the front
    assert result[result["source_id"] == "MolC"]["on_pareto_front"].iloc[0] is False