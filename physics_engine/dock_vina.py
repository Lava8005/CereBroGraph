"""AutoDock Vina Physics Engine for CerebroGraph.

Executes multi-target thermodynamic docking simulations.
Optimized to perform $\mathcal{O}(M)$ RDKit/Meeko 3D embedding (once per molecule)
rather than $\mathcal{O}(M \times T)$. Features real-time CSV appending for 
power-failure resilience and instant pipeline resumption.
"""

import csv
import json
import logging
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError:
    logging.warning("Meeko missing. Fallback strings will be used (expect failures).")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_completed_candidates(output_csv: Path) -> Set[str]:
    """Extract candidate codes already processed to allow seamless resumption."""
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
    """Generate 3D conformer and Meeko PDBQT string exactly once per molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
        
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates using ETKDGv3
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    # Optimize geometry with MMFF94 force field
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass 

    try:
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        # Handle Meeko v0.5+ API changes
        if hasattr(preparator, 'write_pdbqt_string'):
            return preparator.write_pdbqt_string()
        else:
            return PDBQTWriterLegacy.write_string(preparator)[0]
    except Exception as e:
        logging.debug(f"Meeko preparation failed for {smiles}: {e}")
        return None


def dock_precomputed_target(vina_exe: str, ligand_pdbqt_path: str, target: dict) -> Tuple[str, float]:
    """Execute AutoDock Vina using a pre-computed 3D ligand file."""
    disease = target.get("disease", "Unknown")
    receptor_pdbqt = target.get("pdbqt")
    grid = target.get("grid")
    
    if not receptor_pdbqt or not grid or not os.path.exists(receptor_pdbqt):
        return disease, 0.0

    with open(grid, "r", encoding="utf-8") as f:
        g = json.load(f)

    cmd = [
        vina_exe,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt_path,
        "--center_x", str(g["center_x"]),
        "--center_y", str(g["center_y"]),
        "--center_z", str(g["center_z"]),
        "--size_x", str(g["size_x"]),
        "--size_y", str(g["size_y"]),
        "--size_z", str(g["size_z"]),
        "--cpu", "1",
        "--exhaustiveness", "8"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in result.stdout.split("\n"):
            if line.strip().startswith("1"):
                parts = line.split()
                if len(parts) >= 2:
                    return disease, float(parts[1])
    except subprocess.TimeoutExpired:
        logging.warning(f"Vina TIMEOUT for {ligand_pdbqt_path} on {disease}. Terminating.")
    except Exception as e:
        logging.debug(f"Vina crashed for {disease}: {e}")

    return disease, 0.0


def process_molecule(
    vina_exe: str, 
    row: dict, 
    targets: List[dict], 
    work_dir: Path, 
    output_csv: Path, 
    csv_lock: threading.Lock
) -> dict:
    """Worker Thread: Embeds once, docks multi-target, and streams safely to disk."""
    smiles = row.get("smiles", "")
    target_scores = {}
    
    # 1. Embed 3D Conformer ONCE
    ligand_pdbqt_data = prepare_ligand_pdbqt(smiles)
    
    if not ligand_pdbqt_data:
        # Failsafe: Return 0.0 for all targets instantly without I/O
        for t in targets:
            target_scores[t.get("disease", "Unknown")] = 0.0
    else:
        # 2. Write embedded ligand to temporary file safely
        fd, temp_lig_path = tempfile.mkstemp(suffix=".pdbqt", dir=work_dir)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(ligand_pdbqt_data)
                
            # 3. Execute Vina across targets using the shared PDBQT
            for target in targets:
                disease, score = dock_precomputed_target(vina_exe, temp_lig_path, target)
                target_scores[disease] = score
        finally:
            if os.path.exists(temp_lig_path):
                os.remove(temp_lig_path)
                
    row["target_scores"] = json.dumps(target_scores)
    
    # 4. Stream result to CSV immediately (Thread-Safe Append)
    if output_csv:
        with csv_lock:
            file_exists = output_csv.exists()
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                
    return row


def batch_execute_docking(
    input_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    output_csv: Path = Path("data/physics_engine/docking_results.csv"),
    config_path: Path = Path("config/cns_targets.json")
) -> int:
    """Orchestrate resumable multi-threaded docking."""
    if not input_csv.exists() or not config_path.exists():
        logging.error("Missing inputs or target configuration.")
        return 0

    # Locate Vina executable
    vina_exe = "vina" if os.name != "nt" else "vina.exe"
    try:
        subprocess.run([vina_exe, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("AutoDock Vina executable not found in system PATH.")
        return 0

    with open(config_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    # Resumption Logic: Get molecules already processed
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
    
    # Cap threads to prevent CPU thermal throttling (reserve 2 for OS)
    max_threads = max(1, os.cpu_count() - 2)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_molecule, vina_exe, row, targets, work_dir, output_csv, csv_lock): row
            for row in pending_rows
        }
        
        for future in as_completed(futures):
            try:
                future.result()
                total_processed += 1
            except Exception as e:
                logging.error(f"Worker thread crashed: {e}")

    logging.info(f"AutoDock Vina execution completed. Total docked: {total_processed}")
    return total_processed


if __name__ == "__main__":
    batch_execute_docking()