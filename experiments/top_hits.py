"""Step 5 (partial): novelty/identity check of top Pareto hits among unlabeled pilot molecules."""
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
p = pd.read_csv(ROOT / "results/docking_pilot/pool_ranked.csv")
fp = lambda s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
known = p[p.set.isin(["active", "control"])]
kfp = [fp(s) for s in known.smiles]
hits = p[(p.set.isin(["random", "decoy"])) & (p.pareto_rank <= 2)].sort_values(["pareto_rank", "mean_vina"]).head(15).copy()
hits["max_sim_known"] = [max(DataStructs.BulkTanimotoSimilarity(fp(s), kfp)) for s in hits.smiles]
hits["in_B3DB_as_known_compound"] = hits.name.notna() & (hits.name.astype(str).str.len() > 0)
cols = ["name", "pareto_rank", "AChE", "MAOB", "bbb_prob", "bbb_label", "max_sim_known"]
txt = "# Top Pareto hits among unlabeled pilot molecules\n\nAll hits are B3DB compounds (already-known, named molecules), so none is novel by construction; MD and experimental validation were not performed.\n\n" + hits[cols].to_markdown(index=False, floatfmt=".2f")
(ROOT / "results/docking_pilot/top_hits.md").write_text(txt); print(txt)
