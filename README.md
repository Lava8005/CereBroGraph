# OmniVector-AI

OmniVector-AI is an open-source, two-stage computational pipeline designed to screen and rank potential blood-brain-barrier-permeable drug candidates for Alzheimer's disease (specifically targeting Acetylcholinesterase, AChE).

## Overview

The pipeline evaluates a library of candidate small molecules across three critical dimensions:

1. **Target Binding (Physics Engine):** Predicts how well a candidate binds to the AChE target protein (PDB: `1E66`) using docking tools and structure-affinity models (AutoDock Vina / Boltz-2).
2. **BBB Permeability (Vector Engine):** A Graph Neural Network trained on the B3DB dataset predicts the probability that the molecule can physically cross the blood-brain barrier.
3. **Drug-likeness:** Evaluates synthetic accessibility and practical drug-like properties (QED, Lipinski's Rule of Five) using RDKit.

## System Architecture

* **Database (Supabase):** Stores evaluated candidates, their structures, and predicted scores.
* **Ranking Logic:** Computes a multi-objective composite score and calculates the Pareto front to identify the best candidates based on binding, BBB crossing, and safety trade-offs.
* **Dashboard:** A React (Vite) web interface that connects to the database, providing an interactive, sortable ranking table and visual score breakdowns.

## Running the Dashboard

The dashboard is built with React and Vite. To run it locally:

```bash
cd dashboard
npm install
npm run dev
```

Ensure you have created a `.env` file in the `dashboard` directory containing your Supabase connection details:
```env
VITE_SUPABASE_URL=your_project_url
VITE_SUPABASE_ANON_KEY=your_anon_key
```
