import pytest
import sys

def test_supabase_namespace_resolution():
    """Ensure the PyPI library is loaded instead of a shadowed local file."""
    import supabase
    # The __file__ attribute must point to the venv/site-packages directory, NOT the local project
    assert "site-packages" in supabase.__file__ or "dist-packages" in supabase.__file__