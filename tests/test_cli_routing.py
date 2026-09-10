import pytest
from pathlib import Path
from unittest.mock import patch

@patch("urllib.request.urlretrieve")
def test_cli_target_download_routing(mock_retrieve, tmp_path: Path):
    """Ensure CLI routes correctly without hitting RCSB PDB rate limits (403 Forbidden)."""
    
    def mock_download(url, filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text("HEADER    MOCK PDB DATA")
        
    mock_retrieve.side_effect = mock_download
    
    # Imports the correct refactored production function
    from target.prepare_target import fetch_pdb_structure
    
    dummy_dest = tmp_path / "1E66.pdb"
    result = fetch_pdb_structure("1E66", dummy_dest)
    
    assert mock_retrieve.called
    assert result == dummy_dest
    assert dummy_dest.read_text() == "HEADER    MOCK PDB DATA"