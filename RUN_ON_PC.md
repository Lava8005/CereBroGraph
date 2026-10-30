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
dude.docking.org is not reliable for scripted download, so use the mirror on GitHub (this is what worked on the lab PC):
```powershell
$base = "https://raw.githubusercontent.com/ryancoleman/auto-dud-e/master"
$map = [ordered]@{"actives_final.ism"="actives_final.ism"; "decoys_final.ism"="decoys_final.ism"; "rec.pdb"="receptor.pdb"; "xtal-lig.pdb"="xtal-lig.pdb"}
foreach ($t in "aces","aofb") {
  New-Item -ItemType Directory -Force "data\dude\$t" | Out-Null
  foreach ($src in $map.Keys) { Invoke-WebRequest -Uri "$base/$t/$src" -OutFile "data\dude\$t\$($map[$src])" -UseBasicParsing }
}
```
The script now reads `xtal-lig.pdb` directly and fetches the ligand's bond orders from RCSB, so the old mol2 conversion is no longer needed.

**MAO-B keeps its FAD cofactor.** FAD forms part of the MAO-B pocket and Meeko deletes it by default. The code now re-attaches it via Open Babel (`target/cofactors.py`, `keep_cofactors` in `config/cns_targets.json`). After updating, delete old receptors so they are rebuilt:
```powershell
Remove-Item data\target\*.pdbqt -ErrorAction SilentlyContinue
Remove-Item data\physics_engine\docking_results.csv -ErrorAction SilentlyContinue
Remove-Item results\dude_validation -Recurse -ErrorAction SilentlyContinue
python target/prepare_target.py      # log must say: Kept cofactor(s) ['FAD'] ...
```
Then:
```powershell
python -m experiments.dude_validation --targets aces aofb --custom config/validation_targets.json --workers 14 2>&1 | Tee-Object validation.log
```
(Each target needs about 5 minutes for receptor preparation first; the dockings follow.) Output: `results\dude_validation\summary.md`.
