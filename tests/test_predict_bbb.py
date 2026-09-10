import pytest
from pathlib import Path
from vector_engine.predict_bbb import run_bbb_inference

def test_bbb_inference_streaming_bounds(tmp_path: Path):
    """Ensure BBB module streams and returns an integer count, preventing RAM leaks."""
    in_csv = tmp_path / "filtered_step1.csv"
    out_csv = tmp_path / "filtered_step2.csv"
    
    in_csv.write_text("candidate_code,smiles\nOV-000001,CCO\nOV-000002,c1ccccc1\nOV-000003,C\n")
    
    # Run inference with an artificially small chunk size to force batching
    processed_count = run_bbb_inference(
        input_csv=in_csv, 
        output_csv=out_csv, 
        model_weights_path=None, 
        batch_size=2
    )
    
    # Assert return type is the processed integer count, NOT an array
    assert isinstance(processed_count, int)
    assert processed_count == 3
    
    # Verify disk serialization
    assert out_csv.exists()
    lines = out_csv.read_text().strip().split('\n')
    assert len(lines) == 4 # 1 header + 3 candidates
    assert "bbb_probability" in lines[0]