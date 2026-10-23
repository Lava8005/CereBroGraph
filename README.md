# 🧬 CerebroGraph: Polypharmacology Virtual Screening Pipeline

![CI](https://github.com/Lava8005/CereBroGraph/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)
![AutoDock Vina](https://img.shields.io/badge/Physics-AutoDock_Vina-lightgrey.svg)
![React](https://img.shields.io/badge/Dashboard-React-61DAFB.svg)

CerebroGraph is a computationally optimized screening pipeline for CNS (Central Nervous System) polypharmacology: it looks for single molecules that can cross the Blood-Brain Barrier (BBB) **and** bind several disease targets at once.

## 🧠 What and Why?

**The Problem:** Neurological disorders such as Alzheimer's, Parkinson's and Schizophrenia are rarely cured by single-target drugs, and many candidates fail because they never reach the brain.

**The Solution:** CerebroGraph combines a Graph Neural Network for BBB permeability with parallelized AutoDock Vina docking against several receptors, then ranks molecules with multi-objective (Pareto) optimization and publishes the best ones to a React dashboard backed by Supabase.

## ⚙️ How It Works

The pipeline is a cached state machine (`run_pipeline.py`); each stage skips itself when its inputs have not changed.

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 0 | Target preparation: downloads PDB entries, finds the active-site centre from the native ligand, converts receptors to PDBQT | `target/prepare_target.py` | `data/target/` |
| 1 | Library download (ZINC tranches, or a fallback set) + FDA control injection | `library/download_zinc.py` | `data/library/raw/zinc_subset.csv` |
| 2 | Curation: canonical SMILES, InChIKey-based `candidate_code` | `library/curate.py` | `data/library/curated_library.csv` |
| 3 | Physicochemical filter (MW ≤ 500, TPSA ≤ 90, HBD ≤ 3) + QED | `physics_engine/drug_likeness.py` | `data/physics_engine/filtered_step1.csv` |
| 4 | BBB prediction with a GINE graph network (27 node / 7 edge features + 2 global descriptors) | `vector_engine/predict_bbb.py` | `data/physics_engine/filtered_step2.csv` |
| 5 | Multi-target docking with AutoDock Vina (AChE, MAO-B, 5-HT2A) | `physics_engine/dock_vina.py` | `data/physics_engine/docking_results.csv` |
| 6 | Pareto ranking | `ranking/pipeline_ranker.py` | `data/ranking/final_candidates.json` |
| 7 | Sync to Supabase (skipped if credentials are absent) | `database/upload_results.py` | table `polypharmacology_results` |

After ranking, the orchestrator checks which of the five FDA validation controls (donepezil, safinamide, loperamide, risperidone, pimavanserin) reached the Pareto front.

### How candidates are ranked

* A candidate must have a **valid docking score (< 0 kcal/mol) on every target**; otherwise it is dropped (strict polypharmacology).
* Objectives: minimize the Vina score on each target, maximize BBB probability and QED.
* Candidates are ordered by **Pareto rank**, then **crowding distance** (to keep the list diverse; front boundary points get a large finite value so the JSON stays valid), then **mean Vina score**.
* `vina_score` in the output is the **mean** Vina score across targets (more negative is better).
* `composite_score` (0–1, higher is better) is the mean of the min-max-normalized objectives over all docked candidates. It orders molecules *within* a front; it is not what decides Pareto rank.

### Filters added after review

* Permanently charged or multi-fragment molecules (quaternary/aromatic cations, salts) are removed before BBB scoring.
* Candidates need a docking score of at most **-7.0 kcal/mol on every target** (`min_affinity` in `generate_final_ranking`), so weak binders cannot reach the Pareto front just because they are non-dominated on BBB or QED.
* The library provenance (ZINC or the MoleculeNet HIV fallback) is written to `data/library/raw/library_source.json`; report it in any results.

### Known limitation

The BBB model is trained on B3DB and its output is used as-is, with no hard-coded overrides. P-gp substrates such as loperamide can therefore still score above the BBB threshold; the control check reports what the model does, it does not guarantee that controls are filtered out.

---

## 🖥️ Requirements

* **OS:** Linux (Ubuntu 20.04+), macOS, or Windows via WSL2.
* **Python 3.11+** and **Node.js 20.19+** (22 recommended).
* **GPU (optional):** NVIDIA GPU with CUDA for faster GNN training; everything falls back to CPU.
* **RAM:** 16 GB+ recommended.
* **External binaries** (must be on your `PATH`):
  * **AutoDock Vina** (`vina`): <https://vina.scripps.edu/downloads/>
  * **Open Babel** (`obabel`): <https://openbabel.org/docs/current/Installation/install.html> (Ubuntu: `sudo apt install openbabel`)

Check them with `vina --version` and `obabel -V`.

## 🚀 Setup

```bash
# 1. Clone and enter the repository
git clone https://github.com/Lava8005/CereBroGraph.git
cd CereBroGraph

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Supabase (needed for the dashboard and the sync stage)

1. Create a Supabase project.
2. In the SQL editor, run [`db/schema.sql`](db/schema.sql). It is safe to re-run and never drops data.
3. Copy `.env.example` to `.env` in the repository root and fill it in:

   ```bash
   cp .env.example .env
   ```

   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your_service_role_key
   ```

   The service-role key is used **only** by the Python sync script. Keep `.env` out of version control (it is git-ignored).

If you skip this section, the pipeline still runs and simply skips the database sync.

## ▶️ Running the pipeline

```bash
# Build the molecule library (50k molecules including the 5 controls).
python library/download_zinc.py --count 50000
```

By default this downloads from the ZINC tranche list in `ZINC-downloader-2D-smi.uri`. That file is not shipped with the repo: generate it at <https://zinc.docking.org/tranches/home/> (choose the *2D → SMILES* downloader) and place it in the repo root, or pass `--uri-file path/to/file.uri`. **Without it, the script falls back to the public MoleculeNet HIV set** (drug-like, but not ZINC) and says so in the log.

```bash
# Optional: retrain the BBB model (a trained checkpoint is already included).
python vector_engine/train_eval.py

# Run everything: targets -> curation -> filters -> BBB -> docking -> ranking -> DB sync.
python run_pipeline.py
```

* `python run_pipeline.py --force` re-runs every stage and discards previous docking scores.
* Docking is resumable: if it is interrupted, re-running continues from `docking_results.csv`. Scores are discarded automatically if the receptors or grid boxes change.
* To make the database mirror only the latest ranking (deleting rows from older runs), run `python database/upload_results.py --replace` after the pipeline.

## 📊 Dashboard

```bash
cd dashboard
cp .env.example .env       # then set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY (the anon/publishable key)
npm install
npm run dev
```

The dashboard reads the `polypharmacology_results` table (Pareto-front rows only) through Supabase row-level security, which allows public reads and service-role writes. If the two `VITE_` variables are missing, the page shows a configuration message instead of a blank screen.

## ✅ Tests

```bash
python -m pytest              # Python suite (CUDA tests skip automatically without a GPU)
cd dashboard && npm test      # Dashboard (Vitest)
cd dashboard && npm run lint && npm run build
```

GitHub Actions (`.github/workflows/ci.yml`) runs both suites on every push and pull request. Tests that need Supabase credentials are skipped unless `SUPABASE_URL` and `SUPABASE_KEY` are set.

## 🗂️ Repository layout

```
config/           target definitions (PDB IDs and grid-box sizes)
library/          ZINC download + curation
physics_engine/   drug-likeness filter and Vina docking
vector_engine/    B3DB loader, graph featurizer, GINE model, training, inference
ranking/          Pareto sorting and final ranking
database/         Supabase client and upload
db/schema.sql     table + row-level-security policies
dashboard/        React + Vite front end
tests/            pytest suite
```

## 🛠️ Development methodology

Contributions follow three phases: **Architect** (design and trade-offs before code), **TDD Engineer** (tests first, typed and handled errors), and **Reviewer** (complexity, idempotent caching, and security review, e.g. Supabase row-level security).
