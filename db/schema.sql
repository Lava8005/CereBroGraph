create table candidates (
  id uuid primary key default gen_random_uuid(),
  candidate_code text unique not null,     -- e.g. "OV-0001"
  smiles text not null,
  vina_score numeric,          -- raw docking score, kcal/mol, more negative = better binding
  boltz_affinity numeric,      -- filled in later, once Tier-2 exists — nullable for now
  bbb_probability numeric,     -- 0 to 1
  qed numeric,                 -- 0 to 1, drug-likeness
  lipinski_violations int,
  sa_score numeric,
  composite_score numeric,     -- computed by ranking/multi_objective.py
  on_pareto_front boolean default false,   -- computed by ranking/pareto_front.py
  structure_url text,          -- link to a structure file, filled in later
  created_at timestamptz default now()
);
