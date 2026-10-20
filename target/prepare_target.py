"""Target Preparation Module for CerebroGraph.

Automates the acquisition, cleaning, and PDBQT conversion of CNS disease 
protein receptors (AChE, MAO-B, 5-HT2A). Dynamically computes biological 
active-site centroids to ensure thermodynamically valid AutoDock Vina simulations.
"""

import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_receptor_integrity(pdbqt_path: Path) -> bool:
    """Ensure the generated PDBQT file is non-empty and contains valid atomic coordinates."""
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
        
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                return True
    return False


def compute_native_centroid(pdb_path: Path, ligand_names: Set[str]) -> Optional[Tuple[float, float, float]]:
    """Dynamically compute the active site (x, y, z) center using native co-crystallized ligands."""
    if not pdb_path.exists():
        return None
        
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
        
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    
    return cx, cy, cz


def prepare_single_target(target_config: Dict[str, Any], output_dir: Path) -> bool:
    """Prepare receptor PDBQT and grid configuration for a single CNS target."""
    disease = target_config.get("disease", "Unknown")
    pdb_id = target_config.get("pdb_id", "").upper()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = output_dir / f"{pdb_id}_target.pdbqt"
    grid_json = output_dir / f"{pdb_id}_grid.json"
    raw_pdb = output_dir / f"{pdb_id}.pdb"

    # 1. Fetch PDB file from RCSB if missing
    if not raw_pdb.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            logging.info(f"Downloading {pdb_id} from RCSB PDB...")
            urllib.request.urlretrieve(url, raw_pdb)
        except Exception as e:
            logging.error(f"Failed to download PDB {pdb_id}: {e}")
            sys.exit(1) # Hard exit prevents false orchestrator markers

    # 2. Strict Biological Targeting: Extract native ligand centers to align Vina's search space
    ligands_to_extract = {"HUX", "SAG", "RIS"}
    centroid = compute_native_centroid(raw_pdb, ligands_to_extract)
    
    if centroid:
        cx, cy, cz = centroid
    else:
        # Fallback to config file if native ligand is missing from the specific PDB
        active_site = target_config.get("active_site", {})
        cx = float(active_site.get("center_x", 0.0))
        cy = float(active_site.get("center_y", 0.0))
        cz = float(active_site.get("center_z", 0.0))

    # 3. ALWAYS generate Grid Box JSON to overwrite any legacy (0,0,0) stale grids
    box_data = {
        "center_x": round(cx, 3),
        "center_y": round(cy, 3),
        "center_z": round(cz, 3),
        "size_x": float(target_config.get("active_site", {}).get("size_x", 22.0)),
        "size_y": float(target_config.get("active_site", {}).get("size_y", 22.0)),
        "size_z": float(target_config.get("active_site", {}).get("size_z", 22.0))
    }
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(box_data, f, indent=4)

    # 4. Clean PDB and Convert to PDBQT
    if validate_receptor_integrity(receptor_pdbqt):
        logging.info(f"Target {disease} ({pdb_id}) receptor already prepared. Cache hit.")
    else:
        logging.info(f"Preparing {disease} receptor PDBQT ({pdb_id})...")
        cleaned_pdb = output_dir / f"{pdb_id}_clean.pdb"
        
        # Strip both the native ligands AND the structural waters to open the biological pocket
        strip_residues = ligands_to_extract | {"HOH", "WAT", "DUM"}
        
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
            logging.error("Open Babel is required but not found in system PATH. Cannot prepare receptor.")
            sys.exit(1)

        # Validate Open Babel Output
        if not validate_receptor_integrity(receptor_pdbqt):
            logging.error(f"Generated PDBQT for {pdb_id} is empty or invalid.")
            sys.exit(1)

        if cleaned_pdb.exists():
            cleaned_pdb.unlink()

    # Inject absolute paths into the runtime state tracking
    target_config["pdbqt"] = str(receptor_pdbqt.resolve())
    target_config["grid"] = str(grid_json.resolve())
    return True


def prepare_all_targets(config_path: Path = Path("config/cns_targets.json"), output_dir: Path = Path("data/target")):
    """Batch prepare all CNS targets defined in the project configuration."""
    if not config_path.exists():
        logging.error(f"Missing target configuration file: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    resolved_targets = []
    for target in targets:
        if prepare_single_target(target, output_dir):
            resolved_targets.append(target)

    target_index = output_dir / "target_index.json"
    
    # THE FIX: Cache Thrashing Prevention
    # Perform deep dictionary equivalence check. Only rewrite if the OS absolute paths 
    # or the biological coordinates physically changed.
    if target_index.exists():
        try:
            with open(target_index, "r", encoding="utf-8") as f:
                existing_state = json.load(f)
            if existing_state == resolved_targets:
                logging.info("Target index unchanged. Preserving cache timestamps.")
                return
        except json.JSONDecodeError:
            pass # File is corrupted, proceed to overwrite

    # Write state index
    with open(target_index, "w", encoding="utf-8") as f:
        json.dump(resolved_targets, f, indent=4)
    logging.info("Target index updated successfully.")


if __name__ == "__main__":
    prepare_all_targets()