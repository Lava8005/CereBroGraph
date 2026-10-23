# Reproducing the results

Environment: Python 3.13, `pip install -r requirements-lock.txt`; AutoDock Vina 1.1.2 and smina on `PATH`.
All random seeds are fixed (seeds 0-4 for the benchmark, split seed = model seed).

| Step | Command | Output |
|---|---|---|
| 1 BBB benchmark (GNN vs baselines, scaffold + random splits, 5 seeds) | `python -m experiments.bbb_benchmark --split scaffold --seeds 0 1 2 3 4` (and `--split random`), then `python -m experiments.summarize_bbb` | `results/bbb_benchmark/` |
| 2 Pilot targets | `python -m experiments.setup_pilot_targets` | `data/pilot_targets/` |
| 2 Pilot library | `python -m experiments.build_pilot_library` | `results/docking_pilot/library.csv` |
| 2 Docking (resumable, ~2-3 h on 2 CPUs) | `PATH=data/pilot_targets:$PATH python experiments/run_pilot_docking.py` | `results/docking_pilot/docking_results.csv` |
| 2-4 Analysis | `python experiments/analyze_pilot.py` | `results/docking_pilot/analysis.md` |
| Tests | `python -m pytest` | |

Data: B3DB classification table (Meng et al., 2021) in `data/training/`. Receptor structures in the pilot come from the DOCKSTRING bundle; PDB identities were not independently verified.
