import pandas as pd
import numpy as np

def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized 3-objective Pareto identification via sorted sweeping.
    Accepts and returns DataFrames directly to prevent redundant dict serialization.
    """
    if 'vina_score' not in df.columns or len(df) == 0:
        return df

    # Invert vina_score so higher is better across all 3 objectives
    scores = df[['vina_score', 'bbb_probability', 'qed']].copy()
    scores['vina_score'] = -scores['vina_score']
    
    # Convert to NumPy for zero-overhead iteration
    cost_array = scores.values
    is_efficient = np.ones(cost_array.shape[0], dtype=bool)
    
    # Sort by the first objective (vina) descending
    sort_indices = np.argsort(cost_array[:, 0])[::-1]
    
    # The current Pareto front against which to check
    front = []
    
    for idx in sort_indices:
        point = cost_array[idx]
        is_pareto = True
        
        # Check against already found Pareto points (which we know have >= Vina score)
        for front_point in front:
            # If a previously seen point is >= in both BBB and QED, it dominates this point
            # (Strict > is guaranteed because Vina is >= due to the sort order)
            if front_point[1] >= point[1] and front_point[2] >= point[2]:
                is_pareto = False
                break
                
        is_efficient[idx] = is_pareto
        if is_pareto:
            front.append(point)

    df['on_pareto_front'] = is_efficient
    return df