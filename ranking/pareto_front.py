import pandas as pd
import numpy as np

def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the non-dominated Pareto skyline using lexicographical sorting."""
    if df.empty or not {"vina_score", "bbb_probability", "qed"}.issubset(df.columns):
        df["on_pareto_front"] = False
        return df

    # THE FIX: Lexicographical sort to resolve tie-breaker shielding
    df = df.sort_values(
        by=["vina_score", "bbb_probability", "qed"], 
        ascending=[True, False, False]
    ).reset_index(drop=True)

    skyline = []
    
    for i, row in df.iterrows():
        dominated = False
        for front_row in skyline:
            # Check if front_row strictly dominates row
            better_vina = front_row["vina_score"] <= row["vina_score"]
            better_bbb = front_row["bbb_probability"] >= row["bbb_probability"]
            better_qed = front_row["qed"] >= row["qed"]
            
            strictly_better = (
                front_row["vina_score"] < row["vina_score"] or 
                front_row["bbb_probability"] > row["bbb_probability"] or 
                front_row["qed"] > row["qed"]
            )
            
            if better_vina and better_bbb and better_qed and strictly_better:
                dominated = True
                break
                
        if not dominated:
            skyline.append(row)

    skyline_ids = {r.name for r in skyline}
    df["on_pareto_front"] = df.index.isin(skyline_ids)
    return df