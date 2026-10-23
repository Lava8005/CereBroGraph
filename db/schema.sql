-- Safe to re-run: never drops existing data. To reset, run `TRUNCATE polypharmacology_results;` yourself.
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

ALTER TABLE polypharmacology_results ENABLE ROW LEVEL SECURITY;

-- Drop policies before creation so this script can be re-run
DROP POLICY IF EXISTS "Allow public read access" ON polypharmacology_results;
CREATE POLICY "Allow public read access" 
ON polypharmacology_results FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow service role insert and update" ON polypharmacology_results;
-- IMPORTANT: Supabase backend requires the SERVICE_ROLE_KEY to authenticate this policy
CREATE POLICY "Allow service role insert and update" 
ON polypharmacology_results FOR ALL USING (auth.role() = 'service_role');