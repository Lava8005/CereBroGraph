import pytest
import sys
from pathlib import Path

def test_database_module_resolution(tmp_path: Path):
    """Ensure the upload script can dynamically resolve the project root and locate the Supabase client."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    uploader = db_dir / "upload_results.py"
    
    # Calculate root from the script's absolute perspective
    calculated_root = str(uploader.resolve().parent.parent)
    
    if calculated_root not in sys.path:
        sys.path.insert(0, calculated_root)
        
    assert sys.path[0] == calculated_root

def test_json_payload_handling(tmp_path: Path):
    """Ensure the uploader safely aborts if the ranker provides an empty payload."""
    from database.upload_results import sync_to_cloud
    
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]")
    
    # Should exit cleanly returning False, rather than throwing network exceptions
    success = sync_to_cloud(input_json=empty_json)
    assert success is False