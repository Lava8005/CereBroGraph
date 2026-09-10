import pytest
import os
from unittest.mock import patch

def test_supabase_env_validation():
    """Ensure the pipeline crashes loudly if API keys are missing, preventing silent data drops."""
    with patch.dict(os.environ, clear=True):
        from database.supabase import get_client
        with pytest.raises(RuntimeError) as excinfo:
            get_client()
        assert "SUPABASE_URL" in str(excinfo.value)