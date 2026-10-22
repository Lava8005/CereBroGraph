"""TDD Suite for Skyline Pareto Ranking and Queue Truncation."""

import json
import pandas as pd
from pathlib import Path
from ranking.pareto_front import compute_pareto_fronts
from ranking.pipeline_ranker import generate_final_ranking

def test_ranker_pareto_flag_logic():
    """Ensure ranker accurately flags the Pareto front using all mathematical columns."""
    df = pd.DataFrame([
        {"candidate_code": "M1", "source_id": "MolA", "vina_T1": -9.0, "vina_T2": -6.0, "bbb_probability": 0.99, "qed": 0.8},
        {"candidate_code": "M2", "source_id": "MolB", "vina_T1": -6.0, "vina_T2": -9.0, "bbb_probability": 0.40, "qed": 0.5},
        {"candidate_code": "M3", "source_id": "MolC", "vina_T1": -2.0, "vina_T2": -2.0, "bbb_probability": 0.10, "qed": 0.2}
    ])
    
    result = compute_pareto_fronts(df, minimize_cols=["vina_T1", "vina_T2"], maximize_cols=["bbb_probability", "qed"])
    assert "on_pareto_front" in result.columns
    
    mol_a_flag = bool(result[result["source_id"] == "MolA"]["on_pareto_front"].iloc[0])
    mol_c_flag = bool(result[result["source_id"] == "MolC"]["on_pareto_front"].iloc[0])
    
    assert mol_a_flag is True
    assert mol_c_flag is False

def test_generate_final_ranking_truncation(tmp_path: Path):
    """Ensure the ranking pipeline physically truncates the JSON payload to top_k molecules."""
    input_csv = tmp_path / "docking_results.csv"
    output_json = tmp_path / "ranked_candidates.json"
    
    df = pd.DataFrame([
        {
            "candidate_code": f"MOL-{i}", 
            "source_id": f"Mol{i}", 
            "bbb_probability": 0.9, 
            "qed": 0.8,
            "target_scores": '{"Alzheimers_AChE": ' + str(-9.0 + (i*0.1)) + '}'
        }
        for i in range(50)
    ])
    df.to_csv(input_csv, index=False)
    
    generate_final_ranking(input_csv, output_json, top_k=10)
    
    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data) == 10
    assert "on_pareto_front" in data[0]