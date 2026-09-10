import pytest
import pandas as pd
from pathlib import Path
from ranking.pipeline_ranker import generate_final_ranking

def test_top_k_truncation_and_sort_order(tmp_path: Path):
    """Ensure the ranker strictly limits output to K molecules, sorted by optimal metrics."""
    input_csv = tmp_path / "docking_results.csv"
    out_json = tmp_path / "final_candidates.json"
    
    # Generate 50 dummy molecules
    dummy_data = []
    for i in range(50):
        dummy_data.append({
            "candidate_code": f"OV-{i:03d}",
            "smiles": "CCO",
            "vina_score": -5.0 - (i * 0.1),  # Progressively better scores
            "target_scores": "{}"
        })
    pd.DataFrame(dummy_data).to_csv(input_csv, index=False)
    
    # Execute with K=20
    results = generate_final_ranking(input_csv, out_json, top_k=20)
    
    assert len(results) == 20
    # Ensure it grabbed the best (most negative) Vina scores
    assert results[0]["vina_score"] < results[-1]["vina_score"]