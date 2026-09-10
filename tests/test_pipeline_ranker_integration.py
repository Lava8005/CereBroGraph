import pytest
from pathlib import Path
from ranking.pipeline_ranker import generate_final_ranking

def test_pipeline_ranker_uses_real_bbb(tmp_path: Path):
    """Verify ranker prioritizes existing bbb_probability from upstream CSV."""
    test_csv = tmp_path / "pipeline_results.csv"
    test_csv.write_text(
        "candidate_code,smiles,vina_score,qed,lipinski_violations,bbb_probability\n"
        "OV-0001,CCO,-10.5,0.70,0,0.8850\n"
        "OV-0002,c1ccccc1,-9.0,0.60,0,0.4500\n"
    )
    out_json = tmp_path / "final_candidates.json"
    
    results = generate_final_ranking(input_csv=test_csv, output_json=out_json)
    
    assert len(results) == 2
    assert results[0]["bbb_probability"] == 0.8850
    assert results[1]["bbb_probability"] == 0.4500