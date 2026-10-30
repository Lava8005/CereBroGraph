"""Target Preparation Module for CerebroGraph.

Automates the acquisition, cleaning, and PDBQT conversion of CNS disease 
protein receptors (AChE, MAO-B, 5-HT2A). Dynamically computes biological 
active-site centroids to ensure thermodynamically valid AutoDock Vina simulations.
"""
import sys
from pathlib import Path
venv_site_packages = Path(__file__).parent.parent / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages.resolve()))
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from typing import Dict, Any, Optional, Tuple, Set

try:  # works both as `python target/prepare_target.py` and as a package import
    from target.cofactors import (append_to_receptor, cofactor_to_pdbqt_lines, extract_cofactor_lines,
                                  has_residue, protein_lines)
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from target.cofactors import (append_to_receptor, cofactor_to_pdbqt_lines, extract_cofactor_lines,
                                  has_residue, protein_lines)

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
    """Dynamically compute the active site center, strictly isolating the first chain."""
    if not pdb_path.exists():
        return None
        
    coords_by_chain = {}
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() in ligand_names:
                # Column 22 (index 21) contains the Chain Identifier in PDB format
                chain_id = line[21]
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if chain_id not in coords_by_chain:
                        coords_by_chain[chain_id] = []
                    coords_by_chain[chain_id].append((x, y, z))
                except ValueError:
                    continue
                    
    if not coords_by_chain:
        return None
        
    # Pick ONLY the first chain to prevent cross-chain spatial averaging in homodimers
    first_chain = list(coords_by_chain.keys())[0]
    coords = coords_by_chain[first_chain]
    
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

    if not raw_pdb.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            logging.info(f"Downloading {pdb_id} from RCSB PDB...")
            urllib.request.urlretrieve(url, raw_pdb)
        except Exception as e:
            logging.error(f"Failed to download PDB {pdb_id}: {e}")
            sys.exit(1)

    # Native ligand residue codes: E20 = donepezil (4EY7), SAG = safinamide (2V5Z),
    # 8NU = risperidone (6A93). "RIS" is kept only for backward compatibility.
    ligands_to_extract = {"E20", "SAG", "8NU", "RIS"}
    centroid = compute_native_centroid(raw_pdb, ligands_to_extract)
    
    if centroid:
        cx, cy, cz = centroid
    else:
        active_site = target_config.get("active_site", {})
        if not all(k in active_site for k in ("center_x", "center_y", "center_z")):
            logging.error(
                f"No native ligand ({', '.join(sorted(ligands_to_extract))}) found in {pdb_id} and no "
                f"active_site center_x/y/z in config for {disease}. Refusing to dock into a (0,0,0) grid box."
            )
            sys.exit(1)
        logging.warning(f"Native ligand not found in {pdb_id}; using configured active_site center for {disease}.")
        cx = float(active_site["center_x"])
        cy = float(active_site["center_y"])
        cz = float(active_site["center_z"])

    # Generate Grid Box JSON, but ONLY write to disk if coordinates changed
    # to prevent altering the OS mtime and thrashing the docking cache.
    box_data = {
        "center_x": round(cx, 3),
        "center_y": round(cy, 3),
        "center_z": round(cz, 3),
        "size_x": float(target_config.get("active_site", {}).get("size_x", 22.0)),
        "size_y": float(target_config.get("active_site", {}).get("size_y", 22.0)),
        "size_z": float(target_config.get("active_site", {}).get("size_z", 22.0))
    }
    
    write_grid = True
    if grid_json.exists():
        try:
            with open(grid_json, "r", encoding="utf-8") as f:
                existing_box = json.load(f)
            if existing_box == box_data:
                write_grid = False
                logging.info(f"Grid box for {pdb_id} unchanged. Preserving cache.")
        except json.JSONDecodeError:
            pass # File is corrupted, proceed to overwrite

    if write_grid:
        with open(grid_json, "w", encoding="utf-8") as f:
            json.dump(box_data, f, indent=4)

    keep = target_config.get("keep_cofactors", [])
    if validate_receptor_integrity(receptor_pdbqt) and (not keep or has_residue(receptor_pdbqt, keep)):
        logging.info(f"Target {disease} ({pdb_id}) receptor already prepared. Cache hit.")
    else:
        logging.info(f"Preparing {disease} receptor PDBQT ({pdb_id})...")
        cleaned_pdb = output_dir / f"{pdb_id}_clean.pdb"

        # FIX 1: keep protein atoms only (drops waters, glycans, EDO, NO3 and the native ligands, none of
        # which Meeko has templates for). Essential cofactors listed in `keep_cofactors` (FAD for MAO-B) are
        # removed here and re-attached after Meeko, because Meeko would otherwise delete them.
        with open(cleaned_pdb, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(protein_lines(raw_pdb, exclude_resnames=keep)) + "\n")

        try:
            logging.info("Converting clean PDB to PDBQT using Meeko CLI...")
            cmd = [
                sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
                "--read_pdb", str(cleaned_pdb),
                "-o", str(output_dir / f"{pdb_id}_target"),
                "--write_pdbqt", str(receptor_pdbqt),
                "--default_altloc", "A",
                "--delete_bad_res",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # FIX 2: log both streams so the real error is never hidden behind warnings
            if result.returncode != 0:
                logging.error(
                    f"Meeko failed (exit {result.returncode}):\n"
                    f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
                )
                sys.exit(1)

            logging.info(f"Generated target via Meeko: {receptor_pdbqt}")
            if keep:
                cof = extract_cofactor_lines(raw_pdb, keep)
                if not cof:
                    raise RuntimeError(f"Cofactor(s) {keep} not found in {pdb_id}")
                n = append_to_receptor(receptor_pdbqt, cofactor_to_pdbqt_lines(cof))
                logging.info(f"Kept cofactor(s) {keep}: appended {n} atoms to the receptor.")
        except Exception as exc:
            logging.error(f"Receptor conversion failed: {exc}")
            sys.exit(1)

        if not validate_receptor_integrity(receptor_pdbqt):
            logging.error(f"Generated PDBQT for {pdb_id} is empty or invalid.")
            sys.exit(1)

        if cleaned_pdb.exists():
            cleaned_pdb.unlink()

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
    
    # Cache Thrashing Prevention:
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
            pass

    with open(target_index, "w", encoding="utf-8") as f:
        json.dump(resolved_targets, f, indent=4)
    logging.info("Target index updated successfully.")


if __name__ == "__main__":
    prepare_all_targets()