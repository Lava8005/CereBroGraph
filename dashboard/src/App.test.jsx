import { render, screen, waitFor } from '@testing-library/react';
import App from './App';
import { supabase } from './supabase';

// Mock Supabase to prevent network calls during CI/CD
jest.mock('./supabase');

test('renders Pareto skyline candidates from Supabase', async () => {
  supabase.from.mockReturnValue({
    select: jest.fn().mockReturnValue({
      order: jest.fn().mockResolvedValue({
        data: [
          {
            candidate_code: 'MOL-TEST-123',
            source_id: 'DONEPEZIL',
            vina_score: -9.5,
            bbb_probability: 0.98,
            qed: 0.88,
            on_pareto_front: true
          }
        ],
        error: null
      })
    })
  });

  render(<App />);
  
  // Verify Dashboard Header
  expect(screen.getByText(/Polypharmacology Skyline/i)).toBeInTheDocument();
  
  // Verify Payload Data Integration
  await waitFor(() => {
    expect(screen.getByText('MOL-TEST-123')).toBeInTheDocument();
    expect(screen.getByText('DONEPEZIL')).toBeInTheDocument();
    expect(screen.getByText('-9.5')).toBeInTheDocument();
  });
});