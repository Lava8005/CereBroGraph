"""TDD Suite for Supabase Database Synchronization."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from database.upload_results import upload_final_candidates

def test_database_upload_payload_and_upsert(tmp_path: Path):
    """Verify Supabase client is called with EXACT 10-column schema and idempotent upsert."""
    final_json = tmp_path / "final_candidates.json"
    
    # 1. Generate pipeline output mimicking our Ranker
    mock_data = [{
        "candidate_code": "MOL-TEST-123",
        "source_id": "DONEPEZIL",
        "smiles": "CCO",
        "vina_score": -9.5,
        "bbb_probability": 0.98,
        "qed": 0.88,
        "lipinski_violations": 0,
        "composite_score": 0.95,
        "on_pareto_front": True,
        "target_scores": {"Alzheimers_AChE": -9.5}
    }]
    
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    # 2. Patch the Supabase client to prevent actual network HTTP calls
    with patch("database.upload_results.supabase") as mock_supabase:
        # Mock the chained methods: supabase.table().upsert().execute()
        mock_table = MagicMock()
        mock_upsert = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.upsert.return_value = mock_upsert
        
        # 3. Execute
        upload_final_candidates(input_json=final_json)
        
        # 4. Assertions: Verify table target
        mock_supabase.table.assert_called_once_with("polypharmacology_results")
        mock_table.upsert.assert_called_once()
        
        # 5. Assertions: Verify exact payload and UPSERT conflict target
        args, kwargs = mock_table.upsert.call_args
        payload = args[0]
        
        assert len(payload) == 1
        assert payload[0]["candidate_code"] == "MOL-TEST-123"
        assert payload[0]["target_scores"] == {"Alzheimers_AChE": -9.5}
        assert kwargs.get("on_conflict") == "candidate_code"
        
        # THE FIX: Strict 10-column schema enforcement
        expected_keys = {
            "candidate_code", 
            "source_id", 
            "smiles", 
            "vina_score", 
            "bbb_probability", 
            "qed", 
            "lipinski_violations", 
            "composite_score", 
            "on_pareto_front", 
            "target_scores"
        }
        actual_keys = set(payload[0].keys())
        
        # This will fail loudly if ANY key is missing or added
        assert actual_keys == expected_keys, f"Schema mismatch! Missing/Extra keys: {actual_keys.symmetric_difference(expected_keys)}"