import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from physics_engine.dock_vina import batch_execute_docking
d = ROOT / "results/docking_pilot"
batch_execute_docking(d / "library.csv", d / "docking_results.csv", ROOT / "data/pilot_targets/target_index.json",
                      max_workers=2, exhaustiveness=8, timeout=300)
