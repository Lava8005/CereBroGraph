"""Scientific Visualization for Pareto Skyline Analysis."""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pareto_front(json_path: Path = Path("data/ranking/final_candidates.json")):
    if not json_path.exists():
        print(f"Skipping visualization. Missing {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    if df.empty:
        return
        
    # THE FIX: Filter out heavy penalties (99.0) so the plot scale isn't destroyed
    if "vina_score" in df.columns:
        df_valid = df[df["vina_score"] < 0.0].copy()
    else:
        df_valid = df.copy()
        
    if df_valid.empty:
        print("No valid negative affinities to plot.")
        return
    
    pareto_pts = df_valid[df_valid["on_pareto_front"] == True]
    other_pts = df_valid[df_valid["on_pareto_front"] == False]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot dominated molecules
    ax.scatter(other_pts['bbb_probability'], other_pts['qed'], other_pts['vina_score'], 
               c='gray', alpha=0.5, label='Dominated / Sub-optimal')
               
    # Plot Pareto Front
    ax.scatter(pareto_pts['bbb_probability'], pareto_pts['qed'], pareto_pts['vina_score'], 
               c='red', s=60, edgecolors='black', label='Pareto Skyline')
               
    ax.set_xlabel('BBB Probability')
    ax.set_ylabel('QED (Drug-Likeness)')
    ax.set_zlabel('Mean Vina Affinity (kcal/mol)')
    ax.set_title('Multi-Target CNS Polypharmacology Optimization')
    
    plt.legend()
    
    out_file = Path("analysis/pareto_front_3d.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully to {out_file}")

if __name__ == "__main__":
    plot_pareto_front()