"""AutoDock Vina Physics Engine for CerebroGraph.

Executes multi-target thermodynamic docking simulations.
Optimized to perform O(M) RDKit/Meeko 3D embedding (once per molecule)
rather than O(M * T). Features real-time CSV appending and strictly enforced
system exits to preserve the orchestrator's atomic caching state.
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Set, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError as _meeko_err:  # pragma: no cover - depends on environment
    MoleculePreparation = None
    PDBQTWriterLegacy = None
    _MEEKO_ERROR = str(_meeko_err)
else:
    _MEEKO_ERROR = ""

# Vina prints its best pose as the row "   1   <affinity>   <rmsd>   <rmsd>".
_VINA_BEST_POSE = re.compile(r"^\s*1\s+(-?\d+\.\d+)\b")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_completed_candidates(output_csv: Path) -> Set[str]:
    completed = set()
    if output_csv.exists():
        with open(output_csv, "r", encoding="utf-8") as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    if "candidate_code" in row:
                        completed.add(row["candidate_code"])
            except csv.Error:
                pass
    return completed


def prepare_ligand_pdbqt(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
        
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass 

    try:
        preparator = MoleculePreparation()
        prepared = preparator.prepare(mol)
        if isinstance(prepared, (list, tuple)):  # Meeko >= 0.6
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(prepared[0])
            if not is_ok:
                logging.debug(f"Meeko could not write PDBQT for {smiles}: {error_msg}")
                return None
            return pdbqt_string
        return PDBQTWriterLegacy.write_string(preparator)[0]  # Meeko < 0.6
    except Exception as e:
        logging.debug(f"Meeko preparation failed for {smiles}: {e}")
        return None


DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_TIMEOUT_S = 120


def dock_precomputed_target(
    vina_exe: str,
    ligand_pdbqt_path: str,
    target: dict,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Tuple[str, float]:
    """Dock one prepared ligand against one target. Returns (disease, score); 0.0 means failure."""
    disease = target.get("disease", "Unknown")
    receptor_pdbqt = target.get("pdbqt")
    grid = target.get("grid")

    if not receptor_pdbqt or not grid or not os.path.exists(receptor_pdbqt):
        return disease, 0.0

    with open(grid, "r", encoding="utf-8") as f:
        g = json.load(f)

    out_path = f"{ligand_pdbqt_path}.{disease}.out.pdbqt"
    cmd = [
        vina_exe,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt_path,
        "--out", out_path,
        "--center_x", str(g["center_x"]),
        "--center_y", str(g["center_y"]),
        "--center_z", str(g["center_z"]),
        "--size_x", str(g["size_x"]),
        "--size_y", str(g["size_y"]),
        "--size_z", str(g["size_z"]),
        "--cpu", "1",
        "--exhaustiveness", str(exhaustiveness),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for line in result.stdout.split("\n"):
            match = _VINA_BEST_POSE.match(line)
            if match:
                return disease, float(match.group(1))
        logging.debug(f"No pose parsed for {disease}: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        logging.warning(f"Vina TIMEOUT for {ligand_pdbqt_path} on {disease}.")
    except Exception as e:
        logging.debug(f"Vina crashed for {disease}: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    return disease, 0.0


def process_molecule(
    vina_exe: str, 
    row: dict, 
    targets: List[dict], 
    work_dir: Path, 
    output_csv: Path, 
    csv_lock: threading.Lock,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Tuple[dict, bool]:
    smiles = row.get("smiles", "")
    target_scores = {}
    
    ligand_pdbqt_data = prepare_ligand_pdbqt(smiles)
    
    if not ligand_pdbqt_data:
        for t in targets:
            target_scores[t.get("disease", "Unknown")] = 0.0
    else:
        fd, temp_lig_path = tempfile.mkstemp(suffix=".pdbqt", dir=work_dir)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(ligand_pdbqt_data)
                
            for target in targets:
                disease, score = dock_precomputed_target(vina_exe, temp_lig_path, target, exhaustiveness, timeout)
                target_scores[disease] = score
        finally:
            if os.path.exists(temp_lig_path):
                os.remove(temp_lig_path)
                
    row["target_scores"] = json.dumps(target_scores)
    is_success = any(s < 0 for s in target_scores.values())
    
    if output_csv:
        with csv_lock:
            file_exists = output_csv.exists()
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                
    return row, is_success


def batch_execute_docking(
    input_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    output_csv: Path = Path("data/physics_engine/docking_results.csv"),
    config_path: Path = Path("data/target/target_index.json"),
    max_workers: Optional[int] = None,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> int:
    if not input_csv.exists() or not config_path.exists():
        logging.error("Missing inputs or target configuration. Cannot start physics engine.")
        sys.exit(1)

    if MoleculePreparation is None:
        logging.error(
            f"Meeko is required for ligand preparation but could not be imported: {_MEEKO_ERROR}. "
            "Install it with: pip install meeko gemmi"
        )
        sys.exit(1)

    vina_exe = "vina" if os.name != "nt" else "vina.exe"
    try:
        subprocess.run([vina_exe, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("AutoDock Vina executable not found in system PATH.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    invalid = [
        t.get("disease", "Unknown") for t in targets
        if not t.get("pdbqt") or not t.get("grid")
        or not os.path.exists(t["pdbqt"]) or not os.path.exists(t["grid"])
    ]
    if invalid:
        logging.error(f"Receptor or grid file missing for target(s): {', '.join(invalid)}. "
                      "Re-run target/prepare_target.py.")
        sys.exit(1)

    completed_codes = get_completed_candidates(output_csv)
    
    pending_rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("candidate_code") not in completed_codes:
                pending_rows.append(row)

    if not pending_rows:
        logging.info("All candidates already docked. Skipping...")
        return len(completed_codes)

    logging.info(f"Resuming physics engine. {len(pending_rows)} candidates pending...")
    
    work_dir = Path("data/physics_engine/tmp_ligands")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    csv_lock = threading.Lock()
    total_processed = len(completed_codes)
    max_threads = max_workers if max_workers else max(1, (os.cpu_count() or 2) - 2)
    success_count = 0
    failed_per_target = {t.get("disease", "Unknown"): 0 for t in targets}

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_molecule, vina_exe, row, targets, work_dir, output_csv, csv_lock,
                            exhaustiveness, timeout): row
            for row in pending_rows
        }
        
        for future in as_completed(futures):
            try:
                row, is_success = future.result()
                if is_success:
                    success_count += 1
                for disease, score in json.loads(row.get("target_scores", "{}")).items():
                    if score >= 0:
                        failed_per_target[disease] = failed_per_target.get(disease, 0) + 1
                total_processed += 1
            except Exception as e:
                logging.error(f"Worker thread crashed: {e}")

    for disease, failed in failed_per_target.items():
        if failed:
            logging.warning(f"Target {disease}: {failed}/{len(pending_rows)} dockings failed (scored 0.0).")
        if failed == len(pending_rows):
            logging.error(f"Target {disease} failed for EVERY molecule; the strict ranking will be empty. "
                          "Check its receptor and grid box.")

    # Hard exit if 100% of the batch failed, preventing false orchestrator success
    if total_processed > 0 and success_count == 0:
        logging.error("Catastrophic failure: All docked molecules returned 0.0 scores.")
        logging.error("The Vina grid or receptor is likely invalid. Halting pipeline.")
        sys.exit(1)

    logging.info(f"AutoDock Vina execution completed. Total docked: {total_processed} (Successful: {success_count})")
    return total_processed


if __name__ == "__main__":
    # Optional overrides without editing code: CEREBRO_DOCK_WORKERS, CEREBRO_EXHAUSTIVENESS, CEREBRO_TIMEOUT
    batch_execute_docking(
        max_workers=int(os.environ["CEREBRO_DOCK_WORKERS"]) if os.environ.get("CEREBRO_DOCK_WORKERS") else None,
        exhaustiveness=int(os.environ.get("CEREBRO_EXHAUSTIVENESS", DEFAULT_EXHAUSTIVENESS)),
        timeout=int(os.environ.get("CEREBRO_TIMEOUT", DEFAULT_TIMEOUT_S)),
    )