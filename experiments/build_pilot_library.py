"""Build the pilot docking library (Steps 2-5).

Sets
  random   60 BBB+ and 60 BBB- molecules drawn from the seed-0 scaffold TEST split of B3DB
           (never seen by the seed-0 GNN), restricted to docking-friendly sizes.
  active   known AChE / MAO-B ligands (literature knowledge; SMILES checked against known MW).
  decoy    5 property-matched (MW, logP, heavy atoms) test-pool molecules per active.
  control  the remaining FDA validation drugs (loperamide, risperidone, pimavanserin).

Output: results/docking_pilot/library.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from torch_geometric.data import Batch

from vector_engine.b3db_loader import load_b3db_dataset
from vector_engine.b3db_to_graph import smiles_to_graph
from vector_engine.gnn_model import BBBGraphModel

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "results" / "docking_pilot"
SEED = 0
N_PER_CLASS = 60
N_DECOYS_PER_ACTIVE = 5

# name -> (smiles, targets, expected MW). Targets: A = AChE, M = MAO-B.
KNOWN = {
    "donepezil": ("COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "A", 379.5),
    "galantamine": ("CN1CCC23C=CC(O)CC2Oc2c(OC)ccc(C1)c23", "A", 287.4),
    "rivastigmine": ("CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1", "A", 250.3),
    "tacrine": ("Nc1c2c(nc3ccccc13)CCCC2", "A", 198.3),
    "huperzine A": ("CC=C1C2Cc3[nH]c(=O)ccc3C1(N)CC(C)=C2", "A", 242.3),
    "physostigmine": ("CNC(=O)Oc1ccc2c(c1)C1(C)CCN(C)C1N2C", "A", 275.4),
    "selegiline": ("CC(Cc1ccccc1)N(C)CC#C", "M", 187.3),
    "rasagiline": ("C#CCNC1CCc2ccccc12", "M", 171.2),
    "safinamide": ("CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O", "M", 302.3),
    "lazabemide": ("NCCNC(=O)c1ccc(Cl)nc1", "M", 199.6),
    "pargyline": ("C#CCN(C)Cc1ccccc1", "M", 159.2),
    "ladostigil": ("C#CCNC1CCc2cc(OC(=O)N(C)CC)ccc12", "AM", 272.3),
}
CONTROLS = {
    "loperamide": ("CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", 477.0),
    "risperidone": ("Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2", 410.5),
    "pimavanserin": ("CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1", 427.6),
}
EXCLUDE_NAMES = ["donepezil", "galantamine", "rivastigmine", "tacrine", "huperzine", "physostigmine", "selegiline",
                 "deprenyl", "rasagiline", "safinamide", "lazabemide", "pargyline", "ladostigil", "loperamide",
                 "risperidone", "pimavanserin"]
ALLOWED = set("C N O S F Cl Br I P H".split())


def props(smi):
    m = Chem.MolFromSmiles(smi)
    return {
        "mw": Descriptors.MolWt(m), "logp": Descriptors.MolLogP(m), "tpsa": rdMolDescriptors.CalcTPSA(m),
        "hbd": rdMolDescriptors.CalcNumHBD(m), "heavy_atoms": m.GetNumHeavyAtoms(),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(m), "qed": QED.qed(m),
        "charge": sum(a.GetFormalCharge() for a in m.GetAtoms()),
        "elements_ok": all(a.GetSymbol() in ALLOWED for a in m.GetAtoms()),
        "rule_pass": (Descriptors.MolWt(m) <= 500 and rdMolDescriptors.CalcTPSA(m) <= 90 and rdMolDescriptors.CalcNumHBD(m) <= 3),
    }


def gnn_prob(smiles, ckpt):
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    out = []
    for i in range(0, len(smiles), 256):
        graphs = [smiles_to_graph(s, is_inference=True) for s in smiles[i:i + 256]]
        b = Batch.from_data_list(graphs)
        with torch.no_grad():
            out += torch.sigmoid(model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr, global_features=b.global_features)).view(-1).tolist()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # verify literature SMILES against known molecular weights
    for name, (smi, *rest) in {**{k: (v[0], v[2]) for k, v in KNOWN.items()}, **CONTROLS}.items():
        mw = Descriptors.MolWt(Chem.MolFromSmiles(smi))
        expected = rest[-1]
        assert abs(mw - expected) < 0.2, f"{name}: MW {mw:.2f} != expected {expected}"
    print("all reference SMILES match their known molecular weights")

    # seed-0 scaffold test split + names from the raw table
    train, val, test = load_b3db_dataset(split_mode="scaffold", seed=SEED)
    raw = pd.read_csv(ROOT / "data/training/B3DB_classification.tsv", sep="\t")
    name_of = {}
    for smi, nm in zip(raw["SMILES"], raw["compound_name"]):
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            name_of.setdefault(Chem.MolToSmiles(m), str(nm))
    pool = pd.DataFrame({"smiles": [g.smiles for g in test], "bbb_label": [int(g.y.item()) for g in test]})
    pool["name"] = pool["smiles"].map(name_of).fillna("")
    P = pd.DataFrame([props(s) for s in pool["smiles"]])
    pool = pd.concat([pool, P], axis=1)
    lower = pool["name"].str.lower()
    ok = (~lower.apply(lambda n: any(x in n for x in EXCLUDE_NAMES))) & pool["elements_ok"] & (pool["charge"] == 0) \
         & pool["heavy_atoms"].between(10, 32) & (pool["rot_bonds"] <= 7)
    pool = pool[ok].reset_index(drop=True)
    print("docking-friendly test pool:", len(pool), "BBB+:", int(pool.bbb_label.sum()))

    ckpt = ROOT / "results/bbb_benchmark/checkpoints/gine_scaffold_seed0.pt"
    pool["bbb_prob"] = gnn_prob(pool["smiles"].tolist(), ckpt)

    chosen = []
    for lab in (1, 0):
        idx = rng.choice(pool.index[pool.bbb_label == lab], size=N_PER_CLASS, replace=False)
        chosen += list(idx)
    rand = pool.loc[chosen].copy()
    rand["set"] = "random"
    rand["target_label"] = "none"
    remaining = pool.drop(index=chosen)

    rows = []
    # actives
    act_rows = []
    for name, (smi, tl, _) in KNOWN.items():
        m = Chem.MolFromSmiles(smi)
        d = {"smiles": Chem.MolToSmiles(m), "name": name, "set": "active", "target_label": tl, "bbb_label": np.nan}
        d.update(props(smi))
        act_rows.append(d)
    acts = pd.DataFrame(act_rows)
    acts["bbb_prob"] = gnn_prob(acts["smiles"].tolist(), ckpt)

    # matched decoys (nearest by scaled MW / logP / heavy atoms), no reuse
    used, dec = set(), []
    for _, a in acts.iterrows():
        dist = ((remaining["mw"] - a["mw"]).abs() / 50 + (remaining["logp"] - a["logp"]).abs() / 1.0
                + (remaining["heavy_atoms"] - a["heavy_atoms"]).abs() / 4)
        for idx in dist.sort_values().index:
            if idx in used:
                continue
            used.add(idx)
            r = remaining.loc[idx].copy()
            r["set"] = "decoy"
            r["target_label"] = "none"
            r["matched_to"] = a["name"]
            dec.append(r)
            if sum(1 for x in dec if x["matched_to"] == a["name"]) >= N_DECOYS_PER_ACTIVE:
                break
    decoys = pd.DataFrame(dec)

    ctr_rows = []
    for name, (smi, _) in CONTROLS.items():
        m = Chem.MolFromSmiles(smi)
        d = {"smiles": Chem.MolToSmiles(m), "name": name, "set": "control", "target_label": "none", "bbb_label": np.nan}
        d.update(props(smi))
        ctr_rows.append(d)
    ctr = pd.DataFrame(ctr_rows)
    ctr["bbb_prob"] = gnn_prob(ctr["smiles"].tolist(), ckpt)

    lib = pd.concat([rand, acts, decoys, ctr], ignore_index=True)
    lib = lib.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    lib.insert(0, "candidate_code", [f"P{i:04d}" for i in range(len(lib))])
    cols = ["candidate_code", "smiles", "name", "set", "target_label", "matched_to", "bbb_label", "bbb_prob", "qed",
            "mw", "logp", "tpsa", "hbd", "heavy_atoms", "rot_bonds", "rule_pass"]
    lib[cols].to_csv(OUT / "library.csv", index=False)
    print(lib["set"].value_counts().to_dict(), "total", len(lib))
    print("random-set BBB+ rate:", rand.bbb_label.mean(), "| decoys heavy atoms mean %.1f vs actives %.1f"
          % (decoys.heavy_atoms.astype(float).mean(), acts.heavy_atoms.mean()))


if __name__ == "__main__":
    main()
