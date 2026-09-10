import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock Supabase client
vi.mock('./supabase', () => ({
  supabase: {
    from: () => ({
      select: () => ({
        order: () => ({
          limit: vi.fn().mockResolvedValue({
            data: [
              {
                id: '1',
                candidate_code: 'OV-000001',
                composite_score: 0.85,
                bbb_probability: 0.92,
                qed: 0.75,
                target_scores: { "Alzheimers_AChE": -11.2, "Parkinsons_MAOB": -9.8 }
              },
              {
                id: '2',
                candidate_code: 'OV-000002',
                target_scores: {}
              }
            ],
            error: null
          })
        })
      })
    })
  }
}));

describe('App Dashboard Integration', () => {
  it('renders polypharmacology data upon candidate selection', async () => {
    render(<App />);
    
    // Wait for mock data to populate and click the first row
    const firstRow = await screen.findByText('OV-000001');
    fireEvent.click(firstRow);
    
    // Verify JSONB parsing and display
    expect(screen.getByText('Alzheimers AChE')).toBeInTheDocument();
    expect(screen.getByText('-11.20')).toBeInTheDocument();
  });

  it('handles empty target_scores gracefully', async () => {
    render(<App />);
    const secondRow = await screen.findByText('OV-000002');
    fireEvent.click(secondRow);
    
    expect(screen.getByText('No polypharmacology data available.')).toBeInTheDocument();
  });
});