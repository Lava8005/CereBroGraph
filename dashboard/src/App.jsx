import React, { useState, useEffect } from 'react';
import { supabase } from './supabase';
import './App.css';

export default function App() {
  const [candidates, setCandidates] = useState([]);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCandidates() {
      try {
        const { data, error } = await supabase
          .from('candidates')
          .select('*')
          .order('composite_score', { ascending: false })
          .limit(100);
          
        if (error) {
          console.error("Supabase fetch error:", error);
        } else {
          setCandidates(data || []);
          if (data && data.length > 0) {
            setSelectedCandidate(data[0]); // Auto-select the top Pareto candidate
          }
        }
      } catch (err) {
        console.error("Unexpected error fetching data:", err);
      } finally {
        setLoading(false);
      }
    }
    
    fetchCandidates();
  }, []);

  if (loading) {
    return <div className="loading-state">Loading CerebroGraph Triage Data...</div>;
  }

  return (
    <div className="dashboard-container">
      {/* LEFT PANEL: LEADERBOARD TABLE */}
      <div className="table-panel">
        <h2>Pareto Front Candidates</h2>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Composite Rank</th>
                <th>BBB Prob</th>
                <th>QED</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map(c => (
                <tr 
                  key={c.id} 
                  onClick={() => setSelectedCandidate(c)}
                  className={selectedCandidate?.id === c.id ? 'selected-row' : ''}
                >
                  <td><strong>{c.candidate_code}</strong></td>
                  <td>{c.composite_score?.toFixed(4) || 'N/A'}</td>
                  <td>{c.bbb_probability?.toFixed(2) || 'N/A'}</td>
                  <td>{c.qed?.toFixed(2) || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* RIGHT PANEL: POLYPHARMACOLOGY DETAILS */}
      <div className="detail-panel">
        {selectedCandidate ? (
          <div className="detail-card">
            <h3>{selectedCandidate.candidate_code}</h3>
            <p className="smiles-text" title={selectedCandidate.smiles}>
              {selectedCandidate.smiles}
            </p>
            
            <div className="metrics-grid">
              <div className="metric-box">
                <span className="label">Composite Score</span>
                <span className="value">{selectedCandidate.composite_score?.toFixed(4) || '---'}</span>
              </div>
              <div className="metric-box">
                <span className="label">BBB Permeability</span>
                <span className="value">{selectedCandidate.bbb_probability?.toFixed(2) || '---'}</span>
              </div>
              <div className="metric-box">
                <span className="label">Lipinski Violations</span>
                <span className="value">{selectedCandidate.lipinski_violations ?? '---'}</span>
              </div>
            </div>

            <div className="polypharmacology-section">
              <h4>Multi-Target Binding Affinities</h4>
              {selectedCandidate.target_scores && Object.keys(selectedCandidate.target_scores).length > 0 ? (
                <ul className="target-list">
                  {Object.entries(selectedCandidate.target_scores).map(([disease, score]) => (
                    <li key={disease} className="target-item">
                      <span className="disease-name">{disease.replace(/_/g, ' ')}</span>
                      <span className="disease-score">
                        <strong>{score.toFixed(2)}</strong> kcal/mol
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty-state-small">No polypharmacology data available.</div>
              )}
            </div>
            
            {/* Optional 2D Structure Rendering Placeholder */}
            {selectedCandidate.structure_url && (
              <div className="structure-viewer">
                <img 
                  src={selectedCandidate.structure_url} 
                  alt={`2D structure of ${selectedCandidate.candidate_code}`} 
                  loading="lazy"
                />
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state">Select a candidate to view detailed binding affinities.</div>
        )}
      </div>
    </div>
  );
}