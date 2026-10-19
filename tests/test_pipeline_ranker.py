import pandas as pd
from ranking.pareto_front import compute_pareto_front

def test_ranker_truncation_with_bbb_probability():
    """Ensure fixtures contain the required bbb_probability column."""
    df = pd.DataFrame([
        {"source_id": "MolA", "vina_score": -9.0, "bbb_probability": 0.99, "qed": 0.8},
        {"source_id": "MolB", "vina_score": -6.0, "bbb_probability": 0.40, "qed": 0.5}
    ])
    result = compute_pareto_front(df)
    assert "on_pareto_front" in result.columns
    assert result[result["source_id"] == "MolA"]["on_pareto_front"].iloc[0] == True