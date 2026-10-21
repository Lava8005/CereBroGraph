"""TDD Suite for Skyline Pareto Ranking and Queue Truncation."""

import json
import pandas as pd
from pathlib import Path
from ranking.pareto_front import compute_pareto_front
from ranking.pipeline_ranker import generate_final_ranking

def test_ranker_pareto_flag_logic():
    """Ensure ranker accurately flags the Pareto front using all mathematical columns."""
    df = pd.DataFrame([
        {"candidate_code": "M1", "source_id": "MolA", "vina_score": -9.0, "bbb_probability": 0.99, "qed": 0.8, "target_scores": {}},
        {"candidate_code": "M2", "source_id": "MolB", "vina_score": -6.0, "bbb_probability": 0.40, "qed": 0.5, "target_scores": {}},
        {"candidate_code": "M3", "source_id": "MolC", "vina_score": -2.0, "bbb_probability": 0.10, "qed": 0.2, "target_scores": {}}
    ])
    
    result = compute_pareto_front(df)
    assert "on_pareto_front" in result.columns
    
    # THE FIX: Cast the pandas/numpy boolean to standard Python bool to avoid 'is True' pointer mismatch
    mol_a_flag = bool(result[result["source_id"] == "MolA"]["on_pareto_front"].iloc[0])
    mol_c_flag = bool(result[result["source_id"] == "MolC"]["on_pareto_front"].iloc[0])
    
    assert mol_a_flag is True
    assert mol_c_flag is False

def test_generate_final_ranking_truncation(tmp_path: Path):
    """Ensure the ranking pipeline physically truncates the JSON payload to top_k molecules."""
    input_csv = tmp_path / "docking_results.csv"
    output_json = tmp_path / "ranked_candidates.json"
    
    # Generate 50 mock molecules
    df = pd.DataFrame([
        {
            "candidate_code": f"MOL-{i}", 
            "source_id": f"Mol{i}", 
            "vina_score": -9.0 + (i*0.1), 
            "bbb_probability": 0.9, 
            "qed": 0.8,
            "target_scores": "{}"
        }
        for i in range(50)
    ])
    df.to_csv(input_csv, index=False)
    
    # Execute ranker with a strict cutoff of 10
    generate_final_ranking(input_csv, output_json, top_k=10)
    
    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Validation: Must be exactly 10 molecules, properly enriched with derived scores
    assert len(data) == 10
    assert "composite_score" in data[0]
    assert "on_pareto_front" in data[0]