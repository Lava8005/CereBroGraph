import pytest
import pandas as pd
from pathlib import Path
from ranking.pipeline_ranker import generate_final_ranking

def test_empty_csv_handling(tmp_path: Path):
    """Ensure the ranker safely handles 0-byte corrupted cache files."""
    empty_csv = tmp_path / "corrupted_results.csv"
    empty_csv.touch()  # Creates a 0-byte file
    
    out_json = tmp_path / "final.json"
    
    # Execute - should not raise EmptyDataError
    result = generate_final_ranking(input_csv=empty_csv, output_json=out_json)
    
    assert result == []
    assert out_json.exists()
    assert out_json.read_text() == "[]"