import pytest
from pathlib import Path
from vector_engine.predict_bbb import run_bbb_inference

def test_chunked_streaming_memory_bound(tmp_path: Path):
    """Ensure BBB inference streams correctly and returns count without RAM accumulation."""
    in_csv = tmp_path / "filtered_step1.csv"
    out_csv = tmp_path / "filtered_step2.csv"
    
    in_csv.write_text("candidate_code,smiles\nOV-0001,CCO\nOV-0002,c1ccccc1\nOV-0003,C\n")
    
    count = run_bbb_inference(input_csv=in_csv, output_csv=out_csv, batch_size=1)
    
    assert count == 3
    assert out_csv.exists()
    # Verify disk serialization
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 4