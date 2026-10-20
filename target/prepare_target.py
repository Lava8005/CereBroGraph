"""Target Preparation Module for CerebroGraph.

Fetches PDBs, strips native ligands to open the active site, dynamically 
computes the biological centroid of the pocket, and generates Vina-ready PDBQTs.
"""
import json, logging, os, subprocess, sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def compute_native_centroid(pdb_path: Path, ligand_names: set) -> tuple:
    """Dynamically compute the active site (x, y, z) center using native co-crystallized ligands."""
    coords = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() in ligand_names:
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    if not coords:
        return None
    return sum(c[0] for c in coords)/len(coords), sum(c[1] for c in coords)/len(coords), sum(c[2] for c in coords)/len(coords)

def prepare_single_target(target_config: Dict[str, Any], output_dir: Path) -> bool:
    pdb_id = target_config.get("pdb_id", "").upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = output_dir / f"{pdb_id}_target.pdbqt"
    grid_json = output_dir / f"{pdb_id}_grid.json"
    raw_pdb = output_dir / f"{pdb_id}.pdb"

    # 1. Download if missing
    if not raw_pdb.exists():
        import urllib.request
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", raw_pdb)
        except Exception as e:
            logging.error(f"PDB {pdb_id} download failed: {e}")
            sys.exit(1)

    strip_residues = {"HOH", "WAT", "DUM", "HUX", "SAG", "RIS"}

    # 2. Compute true biological active site centroid
    centroid = compute_native_centroid(raw_pdb, strip_residues)
    cx, cy, cz = centroid if centroid else (
        float(target_config.get("active_site", {}).get("center_x", 0.0)),
        float(target_config.get("active_site", {}).get("center_y", 0.0)),
        float(target_config.get("active_site", {}).get("center_z", 0.0))
    )

    # 3. Always write the grid configuration so stale (0,0,0) files are obliterated
    box_data = {
        "center_x": round(cx, 3), "center_y": round(cy, 3), "center_z": round(cz, 3),
        "size_x": float(target_config.get("active_site", {}).get("size_x", 22.0)),
        "size_y": float(target_config.get("active_site", {}).get("size_y", 22.0)),
        "size_z": float(target_config.get("active_site", {}).get("size_z", 22.0))
    }
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(box_data, f, indent=4)

    # 4. Clean PDB and Convert
    if not receptor_pdbqt.exists():
        cleaned_pdb = output_dir / f"{pdb_id}_clean.pdb"
        with open(raw_pdb, "r") as infile, open(cleaned_pdb, "w") as outfile:
            for line in infile:
                if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() in strip_residues:
                    continue
                if line.startswith(("ATOM", "HETATM", "TER", "END")):
                    outfile.write(line)

        obabel_exe = "obabel" if os.name != "nt" else "obabel.exe"
        try:
            subprocess.run([obabel_exe, "-rh", "-xr", str(cleaned_pdb), "-O", str(receptor_pdbqt)], check=True, capture_output=True)
        except Exception:
            logging.error("Open Babel missing or conversion failed.")
            sys.exit(1)
        if cleaned_pdb.exists(): cleaned_pdb.unlink()

    target_config["pdbqt"] = str(receptor_pdbqt.resolve())
    target_config["grid"] = str(grid_json.resolve())
    return True

def prepare_all_targets(config_path: Path = Path("config/cns_targets.json"), output_dir: Path = Path("data/target")):
    with open(config_path, "r", encoding="utf-8") as f:
        targets = json.load(f)
        
    resolved_targets = [target for target in targets if prepare_single_target(target, output_dir)]
    target_index = output_dir / "target_index.json"
    
    # THE FIX: Cache Thrashing Prevention. Only rewrite if values changed.
    if target_index.exists():
        with open(target_index, "r", encoding="utf-8") as f:
            if json.load(f) == resolved_targets:
                return
                
    with open(target_index, "w", encoding="utf-8") as f:
        json.dump(resolved_targets, f, indent=4)

if __name__ == "__main__":
    prepare_all_targets()