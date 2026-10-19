"""TDD Suite for ZINC Library Downloader."""

import pytest
from library.download_zinc import process_smi_line

def test_smi_line_parsing():
    """Ensure various ZINC SMILES delimiters (spaces vs tabs) parse correctly."""
    # Space delimiter
    res1 = process_smi_line("CCO ZINC0000001")
    assert res1["smiles"] == "CCO"
    assert res1["source_id"] == "ZINC0000001"
    
    # Tab delimiter
    res2 = process_smi_line("c1ccccc1\tZINC0000002")
    assert res2["smiles"] == "c1ccccc1"
    assert res2["source_id"] == "ZINC0000002"
    
    # Invalid line / Empty data
    assert process_smi_line("INVALID_LINE_WITHOUT_DELIMITER") is None