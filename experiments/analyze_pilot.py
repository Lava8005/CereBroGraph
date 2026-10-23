"""Steps 2-4 analysis of the pilot docking run. Writes results/docking_pilot/analysis.{md,json}."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from ranking.pareto_front import compute_pareto_fronts

D = ROOT / "results/docking_pilot"
rng = np.random.default_rng(0)
df = pd.read_csv(D / "docking_results.csv")
sc = df["target_scores"].apply(json.loads).apply(pd.Series)
df = pd.concat([df, sc], axis=1)
T = {"AChE": "Alzheimers_AChE", "MAOB": "Parkinsons_MAOB"}
for k, c in T.items():
    df[k] = df[c].where(df[c] < 0)           # 0.0 / missing = failed dock
ok = df.dropna(subset=list(T)).copy()
ok["mean_vina"] = ok[list(T)].mean(axis=1)
out = {"n_library": len(df), "n_docked_both": len(ok), "n_failed": len(df) - len(ok)}
md = ["# Pilot docking analysis (AChE, MAO-B)\n",
      f"Library {len(df)}; successfully docked on both targets: {len(ok)} (failed/timeouts: {len(df)-len(ok)}).\n"]

def boot_auc(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s)
    pt = roc_auc_score(y, s); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        vals.append(roc_auc_score(y[i], s[i]))
    return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ---------- Step 2: enrichment actives vs property-matched decoys ----------
md.append("## Step 2 - Docking validation: known actives vs property-matched decoys\n")
md.append("| Target | n actives | n decoys | AUC (-Vina) | 95% CI | AUC raw size-only baseline (heavy atoms) |\n|---|---|---|---|---|---|")
out["enrichment"] = {}
for k in T:
    lab = "A" if k == "AChE" else "M"
    act = ok[(ok.set == "active") & ok.target_label.str.contains(lab)]
    dec = ok[ok.set == "decoy"]
    d = pd.concat([act, dec]); y = (d.set == "active").astype(int); s = -d[k]
    if y.sum() < 2: continue
    a, lo, hi = boot_auc(y, s)
    base = roc_auc_score(y, d.heavy_atoms)
    out["enrichment"][k] = dict(n_act=int(y.sum()), n_dec=int((1 - y).sum()), auc=a, lo=lo, hi=hi, size_baseline=base)
    md.append(f"| {k} | {y.sum()} | {(1-y).sum()} | {a:.2f} | {lo:.2f}-{hi:.2f} | {base:.2f} |")
md.append("\nActives per target: AChE = donepezil, galantamine, rivastigmine, tacrine, huperzine A, physostigmine, ladostigil; "
          "MAO-B = selegiline, rasagiline, safinamide, lazabemide, pargyline, ladostigil. Decoys are 5 property-matched (MW, logP, heavy atoms) "
          "B3DB test molecules per active, not DUD-E decoys; this is a small pilot, not a benchmark.\n")
r = ok[ok.set.isin(["random", "decoy"])]
out["size_correlation"] = {k: float(np.corrcoef(r.heavy_atoms, r[k])[0, 1]) for k in T}
md.append("Correlation between heavy-atom count and Vina score (random+decoy molecules): " +
          ", ".join(f"{k} r={v:.2f}" for k, v in out["size_correlation"].items()) + " (negative = larger molecules score better, the known Vina size bias).\n")

# per-active ranks within their matched decoys
rows = []
for _, a in ok[ok.set == "active"].iterrows():
    pool = ok[(ok.matched_to == a["name"])]
    for k in T:
        lab = "A" if k == "AChE" else "M"
        if lab in a.target_label and len(pool):
            rows.append((a["name"], k, a[k], int((pool[k] < a[k]).sum()) + 1, len(pool) + 1))
md.append("| Active | Target | Vina | Rank among its matched decoys |\n|---|---|---|---|")
for n, k, v, rk, tot in rows: md.append(f"| {n} | {k} | {v:.1f} | {rk}/{tot} |")
out["per_active"] = [dict(name=n, target=k, vina=v, rank=rk, of=tot) for n, k, v, rk, tot in rows]

# ---------- Step 3: ablation on the random set with TRUE B3DB BBB labels ----------
md.append("\n## Step 3 - Ablation of the selection scheme (random B3DB test molecules, true BBB labels)\n")
R = ok[ok.set == "random"].copy()
K = 20
def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def sel_dock(d): return d.nsmallest(K, "mean_vina")
def sel_rule(d): return d[d.rule_pass].nsmallest(K, "mean_vina")
def sel_gnn(d): return d[d.bbb_prob >= 0.5].nsmallest(K, "mean_vina")
def sel_both(d): return d[d.rule_pass & (d.bbb_prob >= 0.5)].nsmallest(K, "mean_vina")
def sel_ws(d):
    s = norm(-d.mean_vina) + norm(d.bbb_prob) + norm(d.qed); return d.loc[s.nlargest(K).index]
def sel_pareto(d):
    f = compute_pareto_fronts(d.assign(neg=-d.bbb_prob), ["AChE", "MAOB", "neg"], [])
    return d.loc[f.sort_values(["pareto_rank", "crowding_distance"], ascending=[True, False]).head(K).index] if "pareto_rank" in f else d.head(0)
schemes = {"docking only": sel_dock, "docking + physchem rule": sel_rule, "docking + GNN gate (p>=0.5)": sel_gnn,
           "docking + rule + GNN gate": sel_both, "weighted sum (dock, BBB prob, QED)": sel_ws}
try:
    # check pareto API shape
    _t = compute_pareto_fronts(R.assign(neg=-R.bbb_prob).reset_index(drop=True), ["AChE", "MAOB", "neg"], [])
    schemes["Pareto (AChE, MAO-B, BBB prob)"] = lambda d: d.loc[
        compute_pareto_fronts(d.assign(neg=-d.bbb_prob).reset_index(drop=True), ["AChE", "MAOB", "neg"], [])
        .sort_values(["pareto_rank", "crowding_distance"], ascending=[True, False]).head(K).index.map(lambda i: d.index[i])]
except Exception as e:
    md.append(f"(Pareto scheme skipped: {e})\n")
def metrics(sel):
    return dict(n=len(sel), true_bbb_rate=float(sel.bbb_label.mean()) if len(sel) else np.nan,
                mean_vina=float(sel.mean_vina.mean()) if len(sel) else np.nan)
md.append(f"Top-{K} selected from {len(R)} molecules (base rate of true BBB+ = {R.bbb_label.mean():.2f}). 95% bootstrap CIs over molecules.\n")
md.append("| Scheme | n selected | true BBB+ rate | CI | mean Vina | CI |\n|---|---|---|---|---|---|")
out["ablation"] = {}
for name, fn in schemes.items():
    m = metrics(fn(R)); bs_b, bs_v = [], []
    for _ in range(500):
        s = fn(R.iloc[rng.integers(0, len(R), len(R))].reset_index(drop=True))
        if len(s): bs_b.append(s.bbb_label.mean()); bs_v.append(s.mean_vina.mean())
    ci = lambda v: f"{np.percentile(v,2.5):.2f}-{np.percentile(v,97.5):.2f}" if v else "n/a"
    out["ablation"][name] = dict(m, bbb_ci=ci(bs_b), vina_ci=ci(bs_v))
    md.append(f"| {name} | {m['n']} | {m['true_bbb_rate']:.2f} | {ci(bs_b)} | {m['mean_vina']:.2f} | {ci(bs_v)} |")

# ---------- Step 4: retrospective ranking of controls ----------
md.append("\n## Step 4 - Retrospective ranking of known CNS drugs among the pilot library\n")
pool = ok[ok.set.isin(["random", "decoy", "active", "control"])].reset_index(drop=True)
pool["neg"] = -pool.bbb_prob
fr = compute_pareto_fronts(pool, ["AChE", "MAOB", "neg"], [])
pool = pool.join(fr[["pareto_rank"]]) if "pareto_rank" in fr and "pareto_rank" not in pool else pool
pool["dock_rank"] = pool.mean_vina.rank(method="min").astype(int)
md.append(f"Pool size {len(pool)}. Pareto objectives: Vina AChE, Vina MAO-B, BBB probability.\n")
md.append("| Compound | set | Vina AChE | Vina MAO-B | GNN BBB p | rule pass | Pareto rank | rank by mean Vina |\n|---|---|---|---|---|---|---|---|")
out["controls"] = []
for _, r_ in pool[pool.set.isin(["active", "control"])].sort_values("pareto_rank").iterrows():
    md.append(f"| {r_['name']} | {r_['set']} | {r_.AChE:.1f} | {r_.MAOB:.1f} | {r_.bbb_prob:.2f} | {r_.rule_pass} | {int(r_.pareto_rank)} | {r_.dock_rank} |")
    out["controls"].append(dict(name=r_["name"], pareto=int(r_.pareto_rank), dock_rank=int(r_.dock_rank), p=float(r_.bbb_prob)))
md.append(f"\nFraction of pool on Pareto front 1: {(pool.pareto_rank==1).mean():.2f}.")
pool.to_csv(D / "pool_ranked.csv", index=False)
(D / "analysis.md").write_text("\n".join(md)); (D / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
print("\n".join(md))
