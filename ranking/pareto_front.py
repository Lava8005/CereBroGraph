import pandas as pd
import numpy as np

def compute_pareto_fronts(df: pd.DataFrame, minimize_cols: list, maximize_cols: list) -> pd.DataFrame:
    """NSGA-II style non-dominated sorting for true multi-objective optimization."""
    if df.empty or not minimize_cols:
        df["pareto_rank"] = 1
        df["on_pareto_front"] = True
        return df

    data = df.copy()
    # Invert maximize columns so we can treat everything as a minimization problem
    for col in maximize_cols:
        if col in data.columns:
            data[col] = -data[col].astype(float)
            
    # Fallback missing data to heavy penalties
    coords = data[minimize_cols + maximize_cols].fillna(99.0).values
    n = len(coords)
    
    domination_counts = np.zeros(n, dtype=int)
    dominated_lists = [[] for _ in range(n)]
    
    # O(N^2) pairwise dominance check
    for i in range(n):
        for j in range(i+1, n):
            i_dominates_j = np.all(coords[i] <= coords[j]) and np.any(coords[i] < coords[j])
            j_dominates_i = np.all(coords[j] <= coords[i]) and np.any(coords[j] < coords[i])
            
            if i_dominates_j:
                dominated_lists[i].append(j)
                domination_counts[j] += 1
            elif j_dominates_i:
                dominated_lists[j].append(i)
                domination_counts[i] += 1
                
    pareto_ranks = np.zeros(n, dtype=int)
    current_front = [i for i, count in enumerate(domination_counts) if count == 0]
    rank = 1
    
    while current_front:
        next_front = []
        for i in current_front:
            pareto_ranks[i] = rank
            for j in dominated_lists[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front
        rank += 1
        
    df['pareto_rank'] = pareto_ranks
    df['on_pareto_front'] = (pareto_ranks == 1)
    return df