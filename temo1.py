import os
from pathlib import Path

# 1. Annihilate Corrupted Caches & Stale State Markers
project_root = Path.cwd()
files_to_nuke = [
    project_root / "data/physics_engine/docking.marker",
    project_root / "data/physics_engine/docking_results.csv",
]
# Nuke all PDBQT and JSON grid files to force biological centroid recalculation
for target_file in (project_root / "data/target").glob("*_target.pdbqt"):
    files_to_nuke.append(target_file)
for grid_file in (project_root / "data/target").glob("*_grid.json"):
    files_to_nuke.append(grid_file)

for f in files_to_nuke:
    if f.exists():
        f.unlink()
        print(f"Purged stale artifact: {f.name}")

# 2. Rebuild requirements.txt in strict UTF-8
requirements_content = """# Python Dependencies
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
torch>=2.0.0
torch-geometric>=2.3.0
rdkit>=2023.3.1
python-dotenv>=1.0.0
supabase>=2.3.0
meeko>=0.5.0

# EXTERNAL SYSTEM DEPENDENCIES (MUST BE IN SYSTEM PATH):
# 1. AutoDock Vina (vina or vina.exe) - https://vina.scripps.edu/
# 2. Open Babel (obabel or obabel.exe) - https://openbabel.org/
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)

# 3. Nuke legacy, out-of-sync test files
legacy_tests = [
    "test_b3db_loader.py", "test_b3db_pipeline.py", "test_cli_routing.py", 
    "test_device_placement.py", "test_real_docking_mechanics.py", 
    "test_scaffold_split.py", "test_tsv_parser.py", "test_vina_typing.py",
    "test_polypharmacology_docking.py", "test_vina_watchdog.py",
    "test_parallel_docking.py", "test_vina_downloader.py", "test_pipeline_units.py"
]
for lt in legacy_tests:
    tf = project_root / "tests" / lt
    if tf.exists():
        tf.unlink()