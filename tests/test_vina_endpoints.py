import pytest
import urllib.request
import ssl

def test_v1_2_5_artifact_reachability():
    """Ensure the hardcoded artifact URLs point to live, reachable GitHub releases."""
    
    # The verified exact artifact names for v1.2.5
    urls = [
        "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_win.exe",
        "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_x86_64",
        "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for url in urls:
        # Use a HEAD request to avoid downloading the 2MB binaries during testing
        req = urllib.request.Request(url, method="HEAD")
        response = urllib.request.urlopen(req, context=ctx)
        
        # GitHub uses 302 Found to redirect to AWS S3 buckets for the actual binary download
        assert response.status in [200, 302], f"Endpoint {url} returned {response.status}"