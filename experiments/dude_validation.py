"""Validate Vina on DUD-E: (a) enrichment of true actives over decoys, (b) native-ligand re-docking RMSD.

Setup (once, on a machine with internet): for each target download these four files into
  <dude-dir>/<target>/{actives_final.ism, decoys_final.ism, receptor.pdb (or rec.pdb), xtal-lig.pdb (or crystal_ligand.mol2)}
from https://github.com/ryancoleman/auto-dud-e/tree/master/<target>   (aces = AChE, aofb = MAO-B).

Run:  python -m experiments.dude_validation --dude-dir data/dude --targets aces aofb --workers 24
Output: results/dude_validation/{summary.md, summary.json, <target>_scores.csv}
"""
import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
OUT = ROOT / "results" / "dude_validation"
DUDE_COFACTORS = {"aofb": ["FAD"]}   # cofactors that must stay in the receptor
BOX = 22.0  # Angstrom edge of the search box centred on the crystal ligand


# ------------------------------------------------------------------ pure helpers (unit-tested)
def read_ism(path: Path) -> List[Tuple[str, str]]:
    """DUD-E .ism: 'SMILES id [more]' per line -> [(smiles, id)]."""
    rows = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            rows.append((p[0], p[1]))
    return rows


def enrichment_factor(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    """EF at top `frac`: (actives in top fraction / n_top) / (all actives / N). Higher score = better."""
    n = len(y); k = max(1, int(np.ceil(frac * n)))
    top = np.argsort(-score)[:k]
    return float((y[top].sum() / k) / (y.sum() / n)) if y.sum() else float("nan")


def bootstrap_auc(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s)
    pt = roc_auc_score(y, s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


_AD_TO_ELEMENT = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O", "SA": "S", "S": "S",
                  "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "P": "P"}


def parse_pdbqt_pose(text: str) -> Tuple[np.ndarray, List[str]]:
    """Heavy-atom coordinates and elements of the FIRST model of a Vina output PDBQT."""
    xyz, el = [], []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")):
            t = line[77:79].strip().upper() or line.split()[-1].upper()
            if t.startswith("H"):
                continue
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            el.append(_AD_TO_ELEMENT.get(t, t.capitalize()))
    return np.array(xyz), el


def parse_pdbqt_models(text: str):
    """All poses of a Vina output PDBQT as [(xyz, elements), ...], best-scored first."""
    models, cur = [], []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith("ENDMDL"):
            models.append("\n".join(cur)); cur = []
        else:
            cur.append(line)
    return [parse_pdbqt_pose(m + "\nENDMDL") for m in models if m.strip()]


def assignment_rmsd(xyz_a: np.ndarray, el_a: Sequence[str], xyz_b: np.ndarray, el_b: Sequence[str]) -> float:
    """Heavy-atom RMSD with optimal same-element atom matching (symmetry-agnostic, no superposition).

    Slightly optimistic versus graph-isomorphism RMSD because bonds are ignored; report it as such.
    """
    if len(xyz_a) != len(xyz_b) or sorted(el_a) != sorted(el_b):
        return float("nan")
    sq = 0.0
    for e in set(el_a):
        ia = [i for i, x in enumerate(el_a) if x == e]; ib = [i for i, x in enumerate(el_b) if x == e]
        d = ((xyz_a[ia][:, None, :] - xyz_b[ib][None, :, :]) ** 2).sum(axis=2)
        r, c = linear_sum_assignment(d)
        sq += d[r, c].sum()
    return float(np.sqrt(sq / len(xyz_a)))


# ------------------------------------------------------------------ pipeline pieces
def prepare_receptor(pdb: Path, out_pdbqt: Path, keep_cofactors: Sequence[str] = (), ligand_center=None):
    from target.cofactors import append_to_receptor, cofactor_to_pdbqt_lines, extract_cofactor_lines, protein_lines
    clean = out_pdbqt.with_suffix(".clean.pdb")
    clean.write_text("\n".join(protein_lines(pdb, exclude_resnames=keep_cofactors, ligand_center=ligand_center)) + "\n")
    cmd = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor", "--read_pdb", str(clean), "-o",
           str(out_pdbqt.with_suffix("")), "--write_pdbqt", str(out_pdbqt), "--default_altloc", "A", "--delete_bad_res"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_pdbqt.exists():
        raise RuntimeError(f"Meeko receptor preparation failed:\n{r.stderr[-1500:]}")
    if keep_cofactors:
        cof = extract_cofactor_lines(pdb, keep_cofactors)
        if not cof:
            raise RuntimeError(f"Cofactor(s) {list(keep_cofactors)} not found in {pdb}")
        n = append_to_receptor(out_pdbqt, cofactor_to_pdbqt_lines(cof))
        logging.info(f"Kept cofactor(s) {list(keep_cofactors)}: appended {n} atoms to {out_pdbqt.name}")


def ligand_from_xtal_pdb(pdb: Path):
    """DUD-E ships `xtal-lig.pdb` (no bond orders). Coordinates come from the PDB; the SMILES used to prepare the
    ligand for re-docking comes from the RCSB chemical-component dictionary (needs internet)."""
    import urllib.request
    m = Chem.MolFromPDBFile(str(pdb), sanitize=False, removeHs=False)
    if m is None:
        raise RuntimeError(f"Could not read {pdb}")
    res = next(l[17:20].strip() for l in Path(pdb).read_text().splitlines() if l.startswith(("HETATM", "ATOM")))
    info = json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/chemcomp/{res}"))
    desc = info["rcsb_chem_comp_descriptor"]
    smi = desc.get("SMILES") or desc.get("smiles") or desc.get("SMILES_stereo")
    if not smi:
        raise RuntimeError(f"No SMILES for ligand {res} in the RCSB chemical-component record")
    return m, smi


def crystal_ligand(mol2: Path):
    m = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=False)
    if m is None:
        raise RuntimeError(f"Could not read {mol2}")
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def write_grid(mol, grid_json: Path):
    c = mol.GetConformer().GetPositions().mean(axis=0)
    grid_json.write_text(json.dumps({"center_x": float(c[0]), "center_y": float(c[1]), "center_z": float(c[2]),
                                     "size_x": BOX, "size_y": BOX, "size_z": BOX}))


def redock_native(vina_exe: str, target: dict, mol, exhaustiveness: int, timeout: int, smiles: str = None,
                  pose_out: Path = None) -> Dict:
    """Re-dock the crystal ligand. Reports top-1 RMSD, best RMSD among the top 9 poses, and the distance between the
    centroid of the top pose and of the crystal ligand (a large value means Vina chose a different pocket)."""
    from physics_engine.dock_vina import prepare_ligand_pdbqt
    heavy = Chem.RemoveHs(mol, sanitize=False)
    xyz_c = heavy.GetConformer().GetPositions(); el_c = [a.GetSymbol() for a in heavy.GetAtoms()]
    try:
        smi = smiles or Chem.MolToSmiles(Chem.RemoveHs(mol))
        pdbqt = prepare_ligand_pdbqt(smi)
    except Exception:
        pdbqt = None
    if not pdbqt:
        return {"rmsd": float("nan"), "note": "ligand preparation failed"}
    g = json.loads(Path(target["grid"]).read_text())
    with tempfile.TemporaryDirectory() as td:
        lig = Path(td) / "lig.pdbqt"; out = Path(pose_out) if pose_out else Path(td) / "out.pdbqt"
        lig.write_text(pdbqt)
        cmd = [vina_exe, "--receptor", target["pdbqt"], "--ligand", str(lig), "--out", str(out),
               "--center_x", str(g["center_x"]), "--center_y", str(g["center_y"]), "--center_z", str(g["center_z"]),
               "--size_x", str(g["size_x"]), "--size_y", str(g["size_y"]), "--size_z", str(g["size_z"]),
               "--exhaustiveness", str(max(exhaustiveness, 32)), "--num_modes", "9", "--cpu", "8"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        except Exception as e:
            return {"rmsd": float("nan"), "note": f"vina failed: {e}"}
        models = parse_pdbqt_models(out.read_text())
    if not models:
        return {"rmsd": float("nan"), "note": "no pose parsed"}
    rms = [assignment_rmsd(x, e, xyz_c, el_c) for x, e in models]
    c_dist = float(np.linalg.norm(models[0][0].mean(axis=0) - xyz_c.mean(axis=0)))
    finite = [r for r in rms if np.isfinite(r)]
    return {"rmsd": rms[0], "rmsd_best9": min(finite) if finite else float("nan"), "centroid_dist": c_dist,
            "n_heavy": len(el_c), "n_poses": len(models)}


def native_ligand_from_pdb(pdb: Path, resname: str):
    """First copy (first chain/residue) of HETATM residue `resname`, as an RDKit mol (no bond orders)."""
    block, key = [], None
    for line in Path(pdb).read_text().splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == resname:
            k = (line[21], line[22:27])
            key = key or k
            if k == key:
                block.append(line)
    if not block:
        raise RuntimeError(f"No HETATM residue {resname} in {pdb}")
    m = Chem.MolFromPDBBlock("\n".join(block) + "\nEND\n", sanitize=False, removeHs=False)
    if m is None:
        raise RuntimeError(f"Could not parse ligand {resname}")
    return m


def download_pdb(pdb_id: str, dest: Path) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", dest)
    return dest


def matched_decoys(actives: Sequence[str], pool: Sequence[str], per_active: int, seed: int = 0) -> List[str]:
    """Property-matched decoys (MW, logP, heavy atoms, neutral) drawn from `pool`, no reuse."""
    from rdkit.Chem import Descriptors
    props = []
    for smi in pool:
        m = Chem.MolFromSmiles(smi)
        if m is None or Chem.GetFormalCharge(m) != 0 or len(Chem.GetMolFrags(m)) > 1:
            continue
        props.append((smi, Descriptors.MolWt(m), Descriptors.MolLogP(m), m.GetNumHeavyAtoms()))
    P = pd.DataFrame(props, columns=["smi", "mw", "logp", "ha"])
    used, out = set(), []
    for smi in actives:
        m = Chem.MolFromSmiles(smi)
        mw, lp, ha = Descriptors.MolWt(m), Descriptors.MolLogP(m), m.GetNumHeavyAtoms()
        d = (P.mw - mw).abs() / 50 + (P.logp - lp).abs() + (P.ha - ha).abs() / 4
        got = 0
        for i in d.sort_values().index:
            if i not in used and P.smi[i] != smi:
                used.add(i); out.append(P.smi[i]); got += 1
                if got >= per_active:
                    break
    return out


def validate_one(name, work: Path, receptor_pdb: Path, lig_mol, lig_smiles, act, dec, args, keep_cofactors=()):
    from screening.vina_oracle import VinaOracle
    work.mkdir(parents=True, exist_ok=True)
    center = tuple(lig_mol.GetConformer().GetPositions().mean(axis=0))
    prepare_receptor(receptor_pdb, work / "receptor.pdbqt", keep_cofactors, center)
    write_grid(lig_mol, work / "grid.json")
    target = {"disease": name, "pdbqt": str((work / "receptor.pdbqt").resolve()), "grid": str((work / "grid.json").resolve())}
    (work / "target_index.json").write_text(json.dumps([target]))
    res = {}
    if act:
        smi = [s for s, _ in act] + [s for s, _ in dec]
        y = np.array([1] * len(act) + [0] * len(dec))
        oracle = VinaOracle(work / "target_index.json", work / "dock_cache.csv", vina_exe=args.vina, max_workers=args.workers,
                            exhaustiveness=args.exhaustiveness, timeout=args.timeout, work_dir=work / "tmp")
        sc = oracle(smi)[:, 0]
        ok = np.isfinite(sc)
        ha = np.array([Chem.MolFromSmiles(x).GetNumHeavyAtoms() if Chem.MolFromSmiles(x) else 0 for x in smi])
        # Primary: drop failed dockings from BOTH classes. Treating failures as "worst score" would unfairly punish
        # actives, which are larger and more flexible and therefore time out more often.
        y_ok, s_ok = y[ok], -sc[ok]
        auc, lo, hi = bootstrap_auc(y_ok, s_ok)
        score_worst = np.where(ok, -sc, 0.0)
        res = {"n_actives": int(y.sum()), "n_decoys": int((1 - y).sum()), "failed": int((~ok).sum()),
               "failed_actives": int(((~ok) & (y == 1)).sum()), "failed_decoys": int(((~ok) & (y == 0)).sum()),
               "auc": auc, "auc_ci": [lo, hi], "auc_failures_as_worst": float(roc_auc_score(y, score_worst)),
               "ef1": enrichment_factor(y_ok, s_ok, 0.01), "ef5": enrichment_factor(y_ok, s_ok, 0.05),
               "size_only_auc": float(roc_auc_score(y_ok, ha[ok]))}
        pd.DataFrame({"smiles": smi, "label": y, "vina": sc, "heavy_atoms": ha}).to_csv(OUT / f"{name}_scores.csv", index=False)
    res["redock"] = redock_native(args.vina, target, lig_mol, args.exhaustiveness, 900, lig_smiles, work / "redock_pose.pdbqt")
    return res


def report(name, res):
    t = f"## {name}\n\n"
    if "auc" in res:
        t += (f"- actives {res['n_actives']}, decoys {res['n_decoys']}; failed dockings {res['failed']} "
              f"(actives {res['failed_actives']}, decoys {res['failed_decoys']}); metrics below exclude failures\n"
              f"- AUC if failures were scored as worst: {res['auc_failures_as_worst']:.3f}\n"
              f"- ROC-AUC {res['auc']:.3f} (95% CI {res['auc_ci'][0]:.3f}-{res['auc_ci'][1]:.3f}); EF1% {res['ef1']:.1f}; EF5% {res['ef5']:.1f}\n"
              f"- size-only baseline AUC (heavy atoms) {res['size_only_auc']:.3f}  (if close to Vina's, Vina is mostly scoring size)\n")
    rd = res["redock"]
    t += (f"- native-ligand re-docking RMSD: top pose {rd.get('rmsd', float('nan')):.2f} A, best of top {rd.get('n_poses', 0)} "
          f"{rd.get('rmsd_best9', float('nan')):.2f} A, pose-centroid offset {rd.get('centroid_dist', float('nan')):.2f} A "
          f"(success if < 2.0 A; same-element assignment RMSD, slightly optimistic)\n")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dude-dir", default="data/dude")
    ap.add_argument("--targets", nargs="*", default=["aces", "aofb"], help="DUD-E target folders (use none to skip)")
    ap.add_argument("--custom", default=None, help="JSON list of PDB-based targets, e.g. config/validation_targets.json")
    ap.add_argument("--only", nargs="*", default=None, help="restrict --custom to these target names")
    ap.add_argument("--redock-only", action="store_true", help="skip enrichment, only re-dock the native ligands")
    ap.add_argument("--n-actives", type=int, default=100)
    ap.add_argument("--n-decoys", type=int, default=1500)
    ap.add_argument("--decoys-per-active", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--vina", default="vina")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary, md = {}, ["# Validation of the docking setup (enrichment + native re-docking)\n"]

    for t in a.targets or []:
        d = Path(a.dude_dir) / t
        rec = next((d / f for f in ("receptor.pdb", "rec.pdb") if (d / f).exists()), None)
        xt = d / "xtal-lig.pdb"; mol2 = d / "crystal_ligand.mol2"
        miss = [f for f in ("actives_final.ism", "decoys_final.ism") if not (d / f).exists()]
        if rec is None: miss.append("receptor.pdb (or rec.pdb)")
        if not xt.exists() and not mol2.exists(): miss.append("xtal-lig.pdb (or crystal_ligand.mol2)")
        if miss:
            logging.error(f"{t}: missing {miss} in {d}. See the module docstring."); continue
        rng = np.random.default_rng(a.seed)
        act = read_ism(d / "actives_final.ism"); dec = read_ism(d / "decoys_final.ism")
        act = [act[i] for i in rng.permutation(len(act))[: a.n_actives]]
        dec = [dec[i] for i in rng.permutation(len(dec))[: a.n_decoys]]
        if xt.exists():
            lig, lig_smi = ligand_from_xtal_pdb(xt)
        else:
            lig = crystal_ligand(mol2); lig_smi = Chem.MolToSmiles(Chem.RemoveHs(lig))
        keep = DUDE_COFACTORS.get(t, [])
        summary[t] = validate_one(t, OUT / t, rec, lig, lig_smi, act, dec, a, keep)
        md.append(report(t, summary[t]))

    if a.custom:
        pool = [r.split("\t")[0] if "\t" in r else r for r in
                pd.read_csv(ROOT / "data/training/B3DB_classification.tsv", sep="\t")["SMILES"].astype(str)]
        for c in json.loads(Path(a.custom).read_text()):
            name = c["name"]
            if a.only and name not in a.only:
                continue
            pdb = download_pdb(c["pdb_id"], ROOT / "data" / "validation" / f"{c['pdb_id']}.pdb") if not c.get("pdb_file") else Path(c["pdb_file"])
            lig = native_ligand_from_pdb(pdb, c["ligand_resname"])
            act, dec = [], []
            if not a.redock_only and c.get("actives_smi") and Path(c["actives_smi"]).exists():
                act = read_ism(Path(c["actives_smi"]))
                dec = [(s, f"decoy{i}") for i, s in enumerate(matched_decoys([s for s, _ in act], pool, a.decoys_per_active))]
            summary[name] = validate_one(name, OUT / name, pdb, lig, c["ligand_smiles"], act, dec, a, c.get("keep_cofactors", []))
            md.append(report(name, summary[name]))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1)); (OUT / "summary.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
