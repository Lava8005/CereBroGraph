import pytest
from pathlib import Path
import json
from ranking.pipeline_ranker import generate_final_ranking

def test_generate_final_ranking(tmp_path: Path):
    """Ensure the ranker properly merges pipeline metrics and calculates Pareto optimal fronts."""
    # Mock pipeline_results.csv
    mock_csv = tmp_path / "pipeline_results.csv"
    mock_csv.write_text("candidate_code,smiles,vina_score,qed,lipinski_violations\n"
                        "OV-0001,CCO,-9.5,0.74,0\n"
                        "OV-0002,c1ccccc1,-11.2,0.61,0\n")
    
    out_json = tmp_path / "final_candidates.json"
    
    results = generate_final_ranking(mock_csv, out_json)
    
    assert len(results) == 2
    assert "composite_score" in results[0]
    assert "on_pareto_front" in results[0]
    assert "bbb_probability" in results[0]
    assert out_json.exists()