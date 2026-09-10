import os
import pytest

def test_supabase_url_format():
    """Verify SUPABASE_URL conforms to HTTPS and Supabase subdomain conventions if provided."""
    url = os.environ.get("SUPABASE_URL")
    if not url:
        pytest.skip("SUPABASE_URL environment variable is not configured in this shell session.")
        
    assert url.startswith("https://"), "Supabase URL must start with https://"
    assert url.endswith(".supabase.co") or "supabase.co" in url, "URL must point to a valid Supabase domain."

def test_supabase_key_format():
    """Verify SUPABASE_SERVICE_KEY matches either legacy JWT ('eyJ') or modern ('sb_secret_') token formats."""
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not key:
        pytest.skip("SUPABASE_SERVICE_KEY environment variable is not configured in this shell session.")
        
    is_valid_format = key.startswith("eyJ") or key.startswith("sb_secret_")
    assert is_valid_format, "Service key must start with 'sb_secret_' (new format) or 'eyJ' (legacy JWT)."
    assert len(key) >= 20, "Provided Supabase secret key is suspiciously short."