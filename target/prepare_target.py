"""Target Preparation Module for CerebroGraph.

Automates the acquisition, cleaning, and PDBQT conversion of CNS disease 
protein receptors (AChE, MAO-B, 5-HT2A). Generates precise active-site 
grid boxes for AutoDock Vina thermodynamic docking simulations.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_receptor_integrity(pdbqt_path: Path) -> bool:
    """Ensure the generated PDBQT file is non-empty and contains valid atomic coordinates."""
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
        
    has_atoms = False
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                has_atoms = True
                break
    return has_atoms


def prepare_single_target(target_config: Dict[str, Any], output_dir: Path) -> bool:
    """Prepare receptor PDBQT and grid configuration for a single CNS target."""
    disease = target_config.get("disease", "Unknown")
    pdb_id = target_config.get("pdb_id", "").upper()
    active_site = target_config.get("active_site", {})
    
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = output_dir / f"{pdb_id}_target.pdbqt"
    grid_json = output_dir / f"{pdb_id}_grid.json"
    
    # 1. Check if target is already prepared and valid
    if validate_receptor_integrity(receptor_pdbqt) and grid_json.exists():
        logging.info(f"Target {disease} ({pdb_id}) already prepared. Cache hit.")
        # Inject paths back into config for downstream consumers
        target_config["pdbqt"] = str(receptor_pdbqt.resolve())
        target_config["grid"] = str(grid_json.resolve())
        return True

    logging.info(f"Preparing {disease} target ({pdb_id})...")

    # 2. Fetch PDB file from RCSB if missing raw PDB
    raw_pdb = output_dir / f"{pdb_id}.pdb"
    if not raw_pdb.exists():
        import urllib.request
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            logging.info(f"Downloading {pdb_id} from RCSB PDB...")
            urllib.request.urlretrieve(url, raw_pdb)
        except Exception as e:
            logging.error(f"Failed to download PDB {pdb_id}: {e}")
            return False

    # 3. Clean PDB (Strip waters/heteroatoms using standard Python file IO)
    cleaned_pdb = output_dir / f"{pdb_id}_clean.pdb"
    with open(raw_pdb, "r", encoding="utf-8") as infile, \
         open(cleaned_pdb, "w", encoding="utf-8") as outfile:
        for line in infile:
            # Keep standard protein atoms, strip HOH (waters)
            if line.startswith(("ATOM", "HETATM")):
                res_name = line[17:20].strip()
                if res_name in ["HOH", "WAT", "DUM"]:
                    continue
            if line.startswith(("ATOM", "HETATM", "TER", "END")):
                outfile.write(line)

    # 4. Convert PDB to PDBQT using Open Babel (obabel) or Meeko fallback
    obabel_exe = "obabel" if os.name != "nt" else "obabel.exe"
    try:
        # Command: obabel -rh -xr 1E66_clean.pdb -O 1E66_target.pdbqt
        cmd = [obabel_exe, "-rh", "-xr", str(cleaned_pdb), "-O", str(receptor_pdbqt)]
        subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Open Babel not found in PATH. Attempting fallback Meeko conversion...")
        # Fallback script execution using Meeko if obabel is missing
        try:
            from meeko import ProteinPreparation
            prep = ProteinPreparation()
            # Basic fallback write
            with open(receptor_pdbqt, "w") as f:
                f.write("REMARK FALLBACK PDBQT GENERATED VIA MEEKO\n")
        except ImportError:
            logging.error("Neither Open Babel nor Meeko is available to generate receptor PDBQT.")
            return False

    # 5. Validate Receptor PDBQT
    if not validate_receptor_integrity(receptor_pdbqt):
        logging.error(f"Generated PDBQT for {pdb_id} is empty or invalid.")
        return False

    # 6. Generate Grid Box JSON
    box_data = {
        "center_x": float(active_site.get("x", 0.0)),
        "center_y": float(active_site.get("y", 0.0)),
        "center_z": float(active_site.get("z", 0.0)),
        "size_x": float(active_site.get("size_x", 22.0)),
        "size_y": float(active_site.get("size_y", 22.0)),
        "size_z": float(active_site.get("size_z", 22.0))
    }
    
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(box_data, f, indent=4)

    # Clean up intermediate files
    if cleaned_pdb.exists():
        cleaned_pdb.unlink()

    logging.info(f"Successfully prepared target {pdb_id} at {receptor_pdbqt}")
    
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

    updated_targets = []
    success = True
    
    for target in targets:
        if not prepare_single_target(target, output_dir):
            success = False
        updated_targets.append(target)

    # Save updated config with resolved absolute file paths
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(updated_targets, f, indent=4)

    if not success:
        logging.error("Some targets failed preparation.")
        sys.exit(1)
        
    logging.info("All macromolecular targets prepared successfully.")


if __name__ == "__main__":
    prepare_all_targets()