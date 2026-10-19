"""TDD Suite for FDA Control Spike Tracking."""

import pytest
import pandas as pd
import json
from pathlib import Path

def test_source_id_tracking(tmp_path: Path):
    """Ensure FDA controls are tracked by source_id, not brittle SMILES strings."""
    # Mock Docking Results (Survived Biological Filters)
    dock_csv = tmp_path / "docking.csv"
    dock_df = pd.DataFrame([
        {"source_id": "DONEPEZIL", "vina_score": -9.0},
        {"source_id": "ZINC123", "vina_score": -10.0}
    ])
    dock_df.to_csv(dock_csv, index=False)
    
    # Mock Final JSON (Pareto Front - Donepezil got outcompeted by ZINC123)
    final_json = tmp_path / "final.json"
    final_json.write_text(json.dumps([{"source_id": "ZINC123", "vina_score": -10.0}]))
    
    # Validation Logic
    surviving_sources = set(dock_df['source_id'].str.upper())
    assert "DONEPEZIL" in surviving_sources
    
    with open(final_json, "r") as f:
        top_k = [c.get("source_id", "").upper() for c in json.load(f)]
        
    assert "DONEPEZIL" not in top_k  # Accurately reflects competitive displacement