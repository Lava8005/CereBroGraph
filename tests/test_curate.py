import pytest
from pathlib import Path
from library.curate import process_large_library, process_library

def test_curate_backward_compatibility_alias():
    """Verify alias exists for legacy caller imports."""
    assert process_library is process_large_library

def test_large_scale_streaming_output(tmp_path: Path):
    """Ensure curation runs with custom paths and produces canonical 6-digit output."""
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("smiles,zinc_id\nCCO,ZINC1\n")
    out_csv = tmp_path / "curated.csv"
    
    count = process_large_library(input_csv=raw_csv, output_csv=out_csv, batch_size=1)
    
    assert count == 1
    assert "OV-000001" in out_csv.read_text()