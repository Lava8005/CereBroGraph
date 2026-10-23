import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

vi.mock('./supabase', () => {
  const order = vi.fn().mockResolvedValue({
    data: [
      {
        id: 1,
        candidate_code: 'MOL-TEST-123',
        source_id: 'DONEPEZIL',
        vina_score: -9.5,
        bbb_probability: 0.98,
        qed: 0.88,
        composite_score: 0.9,
        on_pareto_front: true,
        target_scores: { Alzheimers_AChE: -10.1, Parkinsons_MAOB: -9.0 },
      },
    ],
    error: null,
  })
  const eq = vi.fn().mockReturnValue({ order })
  const select = vi.fn().mockReturnValue({ eq })
  return { supabase: { from: vi.fn().mockReturnValue({ select }) } }
})

describe('App', () => {
  it('renders Pareto front candidates from Supabase', async () => {
    render(<App />)

    expect(screen.getByText(/Polypharmacology Dashboard/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('MOL-TEST-123')).toBeInTheDocument()
    })
    expect(screen.getByText(/Mean Vina/i)).toBeInTheDocument()
    expect(screen.getByText('DONEPEZIL')).toBeInTheDocument()
    expect(screen.getByText('-9.50')).toBeInTheDocument()
    expect(screen.getByText('98.0%')).toBeInTheDocument()
    expect(screen.getByText(/AChE: -10.1/)).toBeInTheDocument()
  })
})
