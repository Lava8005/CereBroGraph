
CREATE TABLE IF NOT EXISTS polypharmacology_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_code TEXT UNIQUE NOT NULL,
    source_id TEXT,
    smiles TEXT NOT NULL,
    vina_score NUMERIC,
    bbb_probability NUMERIC,
    qed NUMERIC,
    lipinski_violations INT,
    composite_score NUMERIC,
    on_pareto_front BOOLEAN,
    target_scores JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);