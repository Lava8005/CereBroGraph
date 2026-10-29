# Running the 5,000-molecule benchmark on a Windows PC (Ryzen 9 7950X, 64 GB)

Everything below is PowerShell, run from the project folder. Docking uses the CPU only (the GPU does not help Vina).

## 1. One-time setup
```powershell
# unzip CereBroGraph-full.zip, then:
cd CereBroGraph
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Install AutoDock Vina for Windows (https://vina.scripps.edu/downloads/), put `vina.exe` in a folder on PATH, and check `vina --version`.
Open Babel is not needed any more (receptors are prepared with Meeko).
Run `python -m pytest -q` once: expect about 70+ passed, 9 skipped.

## 2. Exhaustive docking of the library (ground truth)
```powershell
python library/download_zinc.py --count 8500     # ~5,000 molecules survive the filters
$env:CEREBRO_DOCK_WORKERS = "24"                  # 16 cores / 32 threads: 24 leaves the PC usable
python run_pipeline.py --force
```
This downloads the receptors, filters, scores BBB, docks every survivor on all targets, and ranks. Docking is resumable: if you stop it, run `python run_pipeline.py` (without `--force`) to continue. Output: `data/physics_engine/docking_results.csv`.

## 3. Compare surrogate-guided screening with random, using that ground truth
```powershell
python -m experiments.surrogate_eval --source csv --csv data/physics_engine/docking_results.csv --seeds 5 --rounds 6
```
Result table: `results/surrogate_eval/csv_summary.md`. Send me that file plus `docking_results.csv`.

## 4. Optional: run the surrogate loop with real docking on a new library
```powershell
python -m screening.run_active --input data/physics_engine/filtered_step2.csv --targets data/target/target_index.json --budget-frac 0.25 --workers 24
```

## 5. DUD-E validation (does Vina enrich real actives on these receptors?)
Download four files per target from https://dude.docking.org/targets/<target>/ into `data/dude/<target>/`:
`actives_final.ism`, `decoys_final.ism`, `receptor.pdb`, `crystal_ligand.mol2` for `aces` (AChE) and `aofb` (MAO-B).
```powershell
python -m experiments.dude_validation --dude-dir data/dude --targets aces aofb --workers 24
```
About 1,600 dockings per target (about 1-2 hours each on your PC). Output: `results/dude_validation/summary.md`. Send it to me.

### 5. (b) The receptors the pipeline really uses, including 5-HT2A
DUD-E has no 5-HT2A, so this step checks the three pipeline structures (4EY7, 2V5Z, 6A93) directly. It downloads each PDB from RCSB, re-docks the native ligand (donepezil, safinamide, risperidone) and reports the RMSD, and for 5-HT2A also docks 12 known 5-HT2A drugs (`data/validation/5ht2a_known_actives.smi`) against property-matched decoys.
```powershell
python -m experiments.dude_validation --targets --custom config/validation_targets.json --workers 24
```
(`--targets` with no names skips the DUD-E part. Run both parts together by giving `--targets aces aofb --custom ...`.)
For a stronger 5-HT2A test, add more actives (SMILES, one per line, `SMILES name`) to that file: ChEMBL target CHEMBL224, potency pChEMBL >= 7, about 100 compounds.
