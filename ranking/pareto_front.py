import pandas as pd
import numpy as np

def compute_pareto_fronts(df: pd.DataFrame, minimize_cols: list, maximize_cols: list) -> pd.DataFrame:
    if df.empty or not minimize_cols:
        df["pareto_rank"] = 1
        df["crowding_distance"] = 0.0
        df["on_pareto_front"] = True
        return df

    data = df.copy()
    for col in maximize_cols:
        if col in data.columns:
            data[col] = -data[col].astype(float)
            
    obj_cols = minimize_cols + maximize_cols
    coords = data[obj_cols].fillna(99.0).values
    n = len(coords)
    
    domination_counts = np.zeros(n, dtype=int)
    dominated_lists = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            i_dom_j = np.all(coords[i] <= coords[j]) and np.any(coords[i] < coords[j])
            j_dom_i = np.all(coords[j] <= coords[i]) and np.any(coords[j] < coords[i])
            if i_dom_j:
                dominated_lists[i].append(j)
                domination_counts[j] += 1
            elif j_dom_i:
                dominated_lists[j].append(i)
                domination_counts[i] += 1
                
    pareto_ranks = np.zeros(n, dtype=int)
    fronts = []
    current_front = [i for i, count in enumerate(domination_counts) if count == 0]
    rank = 1
    
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            pareto_ranks[i] = rank
            for j in dominated_lists[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front
        rank += 1
        
    crowding_distances = np.zeros(n, dtype=float)
    for front in fronts:
        if len(front) <= 2:
            for idx in front:
                crowding_distances[idx] = np.inf
            continue
            
        for obj_idx in range(len(obj_cols)):
            sorted_front = sorted(front, key=lambda x: coords[x, obj_idx])
            obj_min = coords[sorted_front[0], obj_idx]
            obj_max = coords[sorted_front[-1], obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
                
            crowding_distances[sorted_front[0]] = np.inf
            crowding_distances[sorted_front[-1]] = np.inf
                
            for i in range(1, len(sorted_front) - 1):
                if crowding_distances[sorted_front[i]] != np.inf:
                    distance = (coords[sorted_front[i+1], obj_idx] - coords[sorted_front[i-1], obj_idx]) / obj_range
                    crowding_distances[sorted_front[i]] += distance

    df['pareto_rank'] = pareto_ranks
    df['crowding_distance'] = crowding_distances
    df['on_pareto_front'] = (pareto_ranks == 1)
    return df