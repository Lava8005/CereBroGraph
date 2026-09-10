import pytest
import json
from pathlib import Path
from database.upload_results import execute_upload

def test_upload_empty_results_graceful_exit(tmp_path: Path, caplog):
    """Ensure the database sync safely ignores empty ranking outputs without crashing."""
    mock_json = tmp_path / "final_candidates.json"
    mock_json.write_text("[]")
    
    # Execute upload; it should return safely and log a warning, not raise an error
    execute_upload(json_path=mock_json)
    
    assert "0 candidates survived" in caplog.text

def test_upload_missing_file_graceful_exit(tmp_path: Path, caplog):
    """Ensure missing artifacts result in a clean exit."""
    mock_json = tmp_path / "missing.json"
    
    execute_upload(json_path=mock_json)
    
    assert "Pipeline produced no output file" in caplog.text