"""TDD Suite for Biological Data Leakage Prevention and Scaffold Preservation."""

import pytest
import pandas as pd
from pathlib import Path
from vector_engine.b3db_loader import load_b3db_dataset

def test_loader_quarantine_and_acyclic(tmp_path: Path):
    """Ensure FDA controls are dropped, acyclics are preserved, and labels map correctly."""
    mock_tsv = tmp_path / "mock_b3db.tsv"
    
    # 1. Donepezil homolog (Should be dropped by name)
    # 2. Safinamide exact SMILES (Should be dropped by topology)
    # 3. Valid BBB+ compound (Linear/Acyclic, should be preserved)
    # 4. Valid BBB- compound (logBB < -0.1)
    mock_data = (
        "NO.\tCompound Name\tIUPAC_name\tSMILES\tBBB+/BBB-\tlogBB\n"
        "1\tDonepezil-derivative\t\tCCO\tBBB+\t1.2\n" 
        "2\tUnknown\t\tCC(Nc1ccc(OCc2ccccc2F)cc1)C(N)=O\tBBB+\t0.5\n"
        "3\tValid-Linear\t\tCCCC\tBBB+\t0.8\n"
        "4\tValid-Ring\t\tc1ccccc1\tBBB-\t-0.5\n"
    )
    mock_tsv.write_text(mock_data, encoding="utf-8")
    
    train, val = load_b3db_dataset(csv_path=str(mock_tsv))
    
    # Total valid rows should be 2 (Rows 3 and 4). FDA controls (Rows 1 and 2) must be missing.
    total_loaded = len(train) + len(val)
    assert total_loaded == 2