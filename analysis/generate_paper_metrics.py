import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def generate_metrics():
    logging.info("\n" + "="*60)
    logging.info("🔬 CEREBROGRAPH SCIENTIFIC METRICS & ABLATION")
    logging.info("="*60)
    
    try:
        c1 = len(pd.read_csv("data/library/raw/zinc_subset.csv"))
        c2 = len(pd.read_csv("data/library/curated_library.csv"))
        c3 = len(pd.read_csv("data/physics_engine/filtered_step1.csv"))
        c4 = len(pd.read_csv("data/physics_engine/filtered_step2.csv"))
        c5 = len(pd.read_csv("data/physics_engine/docking_results.csv"))
        
        logging.info(f"1. Raw Database Stream:        {c1:,}")
        logging.info(f"2. Curated & Canonicalized:    {c2:,}")
        logging.info(f"3. Rule-based CNS Filter:      {c3:,} ({(c3/c2)*100:.1f}% survival)")
        logging.info(f"4. GNN BBB Permeable Filter:   {c4:,} ({(c4/c3)*100:.1f}% survival)")
        logging.info(f"5. Docked Target Candidates:   {c5:,}")
        logging.info("-" * 60)
    except Exception as e:
        logging.error(f"Failed to generate attrition table: {e}")

    try:
        with open("data/ranking/final_candidates.json", "r") as f:
            candidates = json.load(f)
            
        controls = ["DONEPEZIL", "SAFINAMIDE", "RISPERIDONE", "LOPERAMIDE", "PIMAVANSERIN"]
        df = pd.DataFrame(candidates)
        
        logging.info("🧬 FDA Control Recovery (Redocking / Enrichment):")
        for c in controls:
            match = df[df['source_id'].str.upper() == c]
            if not match.empty:
                rank = match.index[0] + 1
                p_rank = match.iloc[0]['pareto_rank']
                c_dist = match.iloc[0].get('crowding_distance', 0)
                logging.info(f"  - {c}: FOUND! (Global Rank: #{rank} | Pareto Rank: {p_rank} | Divergence: {c_dist:.2f})")
            else:
                logging.info(f"  - {c}: Banished from Skyline or strictly dominated.")
        logging.info("="*60 + "\n")
    except Exception as e:
        logging.error(f"Failed to calculate enrichment metrics: {e}")

if __name__ == "__main__":
    generate_metrics()