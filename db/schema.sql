CREATE TABLE IF NOT EXISTS polypharmacology_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_code TEXT UNIQUE NOT NULL,
    -- ... [Keep existing columns] ...
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- THE FIX: Enable Row Level Security (RLS) to prevent unauthorized mutations
ALTER TABLE polypharmacology_results ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can read the Pareto skyline dashboard
CREATE POLICY "Allow public read access" 
ON polypharmacology_results FOR SELECT USING (true);

-- Policy: Only authenticated service roles (Python Backend) can insert/update
CREATE POLICY "Allow service role insert and update" 
ON polypharmacology_results FOR ALL USING (auth.role() = 'service_role');