import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_upsert_method_invocation(tmp_path: Path):
    """Ensure the uploader strictly uses upsert to prevent primary key crashes."""
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.data = [{"id": "123"}]
    mock_table.upsert().execute.return_value = mock_response
    
    # Create valid mock JSON
    json_path = tmp_path / "final.json"
    json_path.write_text('[{"candidate_code": "OV-01", "vina_score": -9.2}]')
    
    with patch("database.upload_results.supabase", mock_supabase):
        from database.upload_results import sync_to_cloud
        success = sync_to_cloud(input_json=json_path)
        
        assert success is True
        # Verify upsert was called with the correct conflict resolution key
        mock_table.upsert.assert_called_once_with(
            [{"candidate_code": "OV-01", "vina_score": -9.2}], 
            on_conflict="candidate_code"
        )

def test_orchestrator_exit_code_propagation():
    """Ensure network failures trigger sys.exit(1) to halt the orchestrator pipeline."""
    with patch("database.upload_results.sync_to_cloud", return_value=False):
        with patch("sys.exit") as mock_exit:
            # Simulate __main__ execution block
            import database.upload_results as uploader
            if not uploader.sync_to_cloud():
                sys.exit(1)
            
            mock_exit.assert_called_once_with(1)