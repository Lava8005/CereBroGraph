"""TDD Suite for JSONB Payload Parsing and Weighted Multi-Target Score Aggregation."""

import pytest
from ranking.pipeline_ranker import extract_weighted_affinity

def test_vina_score_weighted_aggregation():
    """Ensure the pipeline respects user-defined target weights (e.g., 70% AChE, 30% MAOB)."""
    scores = {"Alzheimers_AChE": -10.0, "Parkinsons_MAOB": -5.0}
    
    # 70% weight to AChE, 30% weight to MAOB
    weights = {"Alzheimers_AChE": 0.7, "Parkinsons_MAOB": 0.3}
    
    weighted_score = extract_weighted_affinity(scores, weights)
    
    # (0.7 * -10.0) + (0.3 * -5.0) = -7.0 + -1.5 = -8.5
    assert weighted_score == -8.5

def test_vina_score_unweighted_fallback():
    """Ensure missing weights default to an equal mean distribution (1/N)."""
    scores = {"AChE": -10.0, "MAOB": -5.0}
    
    # No weights provided; should fallback to mean: (-10 + -5) / 2 = -7.5
    weighted_score = extract_weighted_affinity(scores, {})
    
    assert weighted_score == -7.5