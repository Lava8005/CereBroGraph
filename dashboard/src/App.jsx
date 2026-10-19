import React, { useEffect, useState } from 'react';
import { supabase } from './supabase';
import './App.css';

function App() {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Sorting State
  const [sortConfig, setSortConfig] = useState({ key: 'composite_score', direction: 'desc' });

  useEffect(() => {
    fetchCandidates();
  }, []);

  const fetchCandidates = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // THE FIX: Target correct table and explicitly filter for the Pareto Front
      const { data, error } = await supabase
        .from('polypharmacology_results')
        .select('*')
        .eq('on_pareto_front', true)
        .order('composite_score', { ascending: false });

      if (error) throw error;
      setCandidates(data || []);
    } catch (err) {
      console.error('Error fetching data:', err.message);
      setError('Failed to load polypharmacology candidates. Please check your database connection.');
    } finally {
      setLoading(false);
    }
  };

  // THE FIX: Client-side sorting logic
  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const sortedCandidates = React.useMemo(() => {
    let sortableItems = [...candidates];
    if (sortConfig.key !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [candidates, sortConfig]);

  const getSortIndicator = (columnKey) => {
    if (sortConfig.key === columnKey) {
      return sortConfig.direction === 'asc' ? ' ↑' : ' ↓';
    }
    return '';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Omni-Vector AI: Polypharmacology Dashboard</h1>
        <p>Top {candidates.length} CNS Therapeutics (Pareto Skyline)</p>
      </header>

      <main className="App-main">
        {loading && <div className="loader">Analyzing Multi-Target Affinities...</div>}
        {error && <div className="error-banner">⚠️ {error}</div>}

        {!loading && !error && (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th onClick={() => handleSort('candidate_code')}>Code {getSortIndicator('candidate_code')}</th>
                  <th onClick={() => handleSort('source_id')}>Source ID {getSortIndicator('source_id')}</th>
                  <th onClick={() => handleSort('vina_score')}>Weighted Vina (kcal/mol) {getSortIndicator('vina_score')}</th>
                  <th onClick={() => handleSort('bbb_probability')}>BBB Prob {getSortIndicator('bbb_probability')}</th>
                  <th onClick={() => handleSort('qed')}>QED {getSortIndicator('qed')}</th>
                  <th onClick={() => handleSort('composite_score')}>Composite {getSortIndicator('composite_score')}</th>
                  <th>Target Breakdown</th>
                </tr>
              </thead>
              <tbody>
                {sortedCandidates.map((c) => (
                  <tr key={c.id || c.candidate_code}>
                    <td><strong>{c.candidate_code}</strong></td>
                    <td className={['DONEPEZIL', 'SAFINAMIDE'].includes(c.source_id) ? 'control-highlight' : ''}>
                      {c.source_id}
                    </td>
                    <td className="score-metric">{c.vina_score?.toFixed(2)}</td>
                    <td className="score-metric">{(c.bbb_probability * 100).toFixed(1)}%</td>
                    <td className="score-metric">{c.qed?.toFixed(3)}</td>
                    <td className="score-metric composite">{c.composite_score?.toFixed(3)}</td>
                    <td className="target-scores-cell">
                      {c.target_scores ? (
                        <div className="target-pills">
                          {Object.entries(c.target_scores).map(([target, score]) => (
                            <span key={target} className="target-pill" title={target}>
                              {target.split('_')[1]}: {score.toFixed(1)}
                            </span>
                          ))}
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </td>
                  </tr>
                ))}
                {sortedCandidates.length === 0 && (
                  <tr>
                    <td colSpan="7" style={{textAlign: "center"}}>No candidates found on the Pareto front.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;