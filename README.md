# 🧬 CerebroGraph: Polypharmacology Virtual Screening Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)
![AutoDock Vina](https://img.shields.io/badge/Physics-AutoDock_Vina-lightgrey.svg)
![React](https://img.shields.io/badge/Dashboard-React-61DAFB.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

CerebroGraph is a high-throughput, computationally optimized bioinformatics pipeline designed for CNS (Central Nervous System) polypharmacology. 

## 🧠 What and Why?

**The Problem (Why):** Neurological disorders like Alzheimer's, Parkinson's, and Schizophrenia are rarely cured by single-target drugs. Traditional drug discovery targets one protein at a time, often resulting in therapies that fail to cross the Blood-Brain Barrier (BBB) or lack the multi-faceted efficacy needed for complex brain diseases.

**The Solution (What):** CerebroGraph utilizes **Polypharmacology**—discovering single molecules that purposefully target multiple disease receptors simultaneously. By orchestrating Graph Neural Networks (GNNs) for BBB permeability prediction and parallelized thermodynamic docking (AutoDock Vina), CerebroGraph mathematically isolates the most promising multi-target therapeutic candidates from massive chemical libraries.

## ⚙️ How It Works: System Architecture

The pipeline operates as a strictly cached, idempotent state machine across five tiers:
1. **Data Ingestion & Curation:** Canonicalizes SMILES and applies deterministic InChIKey hashing for collision-free relational database synchronization.
2. **Physicochemical Filtering:** Strict CNS multi-parameter optimization ($MW \le 500$, $TPSA \le 90$, $HBD \le 3$).
3. **Machine Learning (Vector Engine):** A 27/7/2 Graph Isomorphism Network with Edges (GINE) predicts BBB permeability using topological embeddings and global molecular descriptors.
4. **Physics Engine:** Automates RCSB PDB retrieval, isolates biological active site centroids (chain-aware), and executes $\mathcal{O}(M)$ parallelized AutoDock Vina simulations across multiple receptors (AChE, MAO-B, 5-HT2A).
5. **Pareto Skyline Ranking:** Utilizes multi-objective optimization to rank molecules and syncs the Pareto front to a Supabase-backed React dashboard.

---

## 🛠️ The 3-in-1 Development Methodology

CerebroGraph is built and maintained using a strict **3-in-1 Full-Cycle Software Development Pipeline**. All future contributions must adhere to these three sequential phases:

1. **🏛️ ARCHITECT (System Design & Trade-offs):** Before writing code, analyze scale, memory, and dependencies. Propose architectural approaches, weigh trade-offs (e.g., cross-chain centroid averaging vs. isolated chain targeting), and lock in a blueprint.
2. **🧪 TDD ENGINEER (Test-Driven Implementation):** Write comprehensive test suites FIRST (e.g., catching FDA controls leaking through homologous naming). Provide robust, production-ready code that passes the suite with structured error handling and strict typing.
3. **🔍 REVIEWER (Strict Code & Security Audit):** Conduct Big-O complexity analysis, ensure idempotent state-machine caching (preventing cache-thrashing via OS `st_mtime`), and audit security (e.g., Supabase Row Level Security).

---

## 🖥️ System Requirements & Prerequisites

### 1. Hardware
* **OS:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows (WSL2 preferred).
* **GPU (Recommended):** NVIDIA GPU with CUDA 12.x support for GNN training. Will seamlessly fall back to CPU.
* **RAM:** 16GB+ Minimum.

### 2. External System Binaries (CRITICAL)
CerebroGraph requires two external C++ cheminformatics tools to be installed and accessible in your system's `PATH`:
* **AutoDock Vina** (`vina` or `vina.exe`): [Download Here](https://vina.scripps.edu/downloads/)
* **Open Babel** (`obabel` or `obabel.exe`): [Download Here](https://openbabel.org/docs/current/Installation/install.html) (On Ubuntu: `sudo apt install openbabel`)

*Verification:* Run `vina --version` and `obabel -V` in your terminal. Both must return valid version numbers before proceeding.

### 3. Environment Setup
Ensure Python 3.11+ and Node.js are installed.
```bash
# Clone the repository
git clone [https://github.com/your-username/CerebroGraph.git](https://github.com/your-username/CerebroGraph.git)
cd CerebroGraph

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install strictly defined dependencies
pip install -r requirements.txt

Steps to Run -:
# .env (Place in root directory)
SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
SUPABASE_KEY=your_service_role_key

python library/download_zinc.py --count 50000

python vector_engine/train_eval.py

python run_pipeline.py

cd dashboard
npm install
npm run dev