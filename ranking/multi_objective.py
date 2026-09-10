import pandas as pd

def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the composite score directly on DataFrames to prevent serialization lag.
    Ranking formula: composite_score = 0.5 * normalized(binding) + 0.35 * bbb + 0.15 * qed
    """
    if 'vina_score' not in df.columns or len(df) <= 1:
        if len(df) == 1:
            df['composite_score'] = 0.5 * 1.0 + 0.35 * df.get('bbb_probability', 0) + 0.15 * df.get('qed', 0)
        return df

    # Vina score: more negative = better. Rank 1 is the best.
    df['vina_rank'] = df['vina_score'].rank(method='min', ascending=True)
    
    # Normalize to 0-1 percentile
    n = len(df)
    df['norm_vina'] = (n - df['vina_rank']) / (n - 1)
    
    # Calculate composite score
    df['composite_score'] = (
        0.5 * df['norm_vina'] + 
        0.35 * df['bbb_probability'] + 
        0.15 * df['qed']
    ).round(4)
    
    return df.drop(columns=['vina_rank', 'norm_vina'])