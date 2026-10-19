import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def validate_receptor_integrity(pdbqt_path: Path) -> bool:
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                return True
    return False

def prepare_single_target(target_config: Dict[str, Any], output_dir: Path) -> bool:
    disease = target_config.get("disease", "Unknown")
    pdb_id = target_config.get("pdb_id", "").upper()
    active_site = target_config.get("active_site", {})
    
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = output_dir / f"{pdb_id}_target.pdbqt"
    grid_json = output_dir / f"{pdb_id}_grid.json"
    
    if validate_receptor_integrity(receptor_pdbqt) and grid_json.exists():
        logging.info(f"Target {disease} ({pdb_id}) already prepared.")
        target_config["pdbqt"] = str(receptor_pdbqt.resolve())
        target_config["grid"] = str(grid_json.resolve())
        return True

    raw_pdb = output_dir / f"{pdb_id}.pdb"
    if not raw_pdb.exists():
        import urllib.request
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            logging.info(f"Downloading {pdb_id} from RCSB PDB...")
            urllib.request.urlretrieve(url, raw_pdb)
        except Exception as e:
            logging.error(f"Failed to download PDB {pdb_id}: {e}")
            sys.exit(1) # Strict failure, no mocks

    cleaned_pdb = output_dir / f"{pdb_id}_clean.pdb"
    # THE FIX: Explicitly strip native ligands blocking the active site
    strip_residues = {"HOH", "WAT", "DUM", "HUX", "SAG", "RIS"}
    
    with open(raw_pdb, "r", encoding="utf-8") as infile, \
         open(cleaned_pdb, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                res_name = line[17:20].strip()
                if res_name in strip_residues:
                    continue
            if line.startswith(("ATOM", "HETATM", "TER", "END")):
                outfile.write(line)

    obabel_exe = "obabel" if os.name != "nt" else "obabel.exe"
    try:
        cmd = [obabel_exe, "-rh", "-xr", str(cleaned_pdb), "-O", str(receptor_pdbqt)]
        subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Open Babel is required but not found in PATH.")
        sys.exit(1)

    if not validate_receptor_integrity(receptor_pdbqt):
        logging.error(f"Generated PDBQT for {pdb_id} is empty or invalid.")
        sys.exit(1)

    # THE FIX: Correct coordinate keys mapping to cns_targets.json
    box_data = {
        "center_x": float(active_site.get("center_x", 0.0)),
        "center_y": float(active_site.get("center_y", 0.0)),
        "center_z": float(active_site.get("center_z", 0.0)),
        "size_x": float(active_site.get("size_x", 22.0)),
        "size_y": float(active_site.get("size_y", 22.0)),
        "size_z": float(active_site.get("size_z", 22.0))
    }
    
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(box_data, f, indent=4)

    if cleaned_pdb.exists(): cleaned_pdb.unlink()
    
    target_config["pdbqt"] = str(receptor_pdbqt.resolve())
    target_config["grid"] = str(grid_json.resolve())
    return True

def prepare_all_targets(config_path: Path = Path("config/cns_targets.json"), output_dir: Path = Path("data/target")):
    if not config_path.exists():
        logging.error(f"Missing config: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    resolved_targets = []
    for target in targets:
        prepare_single_target(target, output_dir)
        resolved_targets.append(target)

    # THE FIX: Write to state index, NOT to the tracked config file
    target_index = output_dir / "target_index.json"
    with open(target_index, "w", encoding="utf-8") as f:
        json.dump(resolved_targets, f, indent=4)

if __name__ == "__main__":
    prepare_all_targets()