import pytest
import urllib.request
import ssl
from physics_engine.dock_vina import ensure_vina_binary

def test_v1_2_5_artifact_urls(monkeypatch):
    """Ensure the hardcoded URLs point to live, reachable GitHub release artifacts."""
    # Test the Windows URL specifically
    windows_url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_windows_x86_64.exe"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # We only need to request the headers to verify it's a 200 OK or 302 Redirect, avoiding a full download
    req = urllib.request.Request(windows_url, method="HEAD")
    response = urllib.request.urlopen(req, context=ctx)
    
    assert response.status in [200, 302]