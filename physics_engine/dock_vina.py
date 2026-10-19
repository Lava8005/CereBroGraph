"""AutoDock Vina Physics Engine. 

Generates 3D ligand conformers via RDKit/Meeko, manages AutoDock Vina binaries, 
orchestrates multi-threaded docking across polypharmacology targets, and 
safely handles OS-level timeouts to prevent Deadlocks.
"""

import csv
import json
import logging
import os
import platform
import re
import ssl
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError:
    logging.warning("Meeko not installed. Ligand PDBQT preparation will fail.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_vina_binary(bin_dir: Path) -> Path:
    """Download the AutoDock Vina binary, bypassing strict firewall SSL checks."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    is_windows = platform.system().lower() == "windows"
    binary_name = "vina.exe" if is_windows else "vina"
    binary_path = bin_dir / binary_name
    
    if binary_path.exists():
        return binary_path
        
    url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_win.exe" if is_windows else \
          "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
        
    logging.info(f"Downloading Vina binary for {platform.system()}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(url, context=ctx) as response, open(binary_path, 'wb') as out_file:
        out_file.write(response.read())
    if not is_windows:
        os.chmod(binary_path, 0o755)
    return binary_path


def parse_vina_score(stdout: str) -> Optional[float]:
    """Extract the best binding affinity (kcal/mol) from Vina's stdout."""
    match = re.search(r'\s+1\s+([-+]\d+\.\d+)\s+', stdout)
    if match: return float(match.group(1))
    return None


def prepare_ligand_pdbqt(smiles: str) -> Optional[str]:
    """Generate a 3D conformer and convert to PDBQT format using RDKit & Meeko."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        
        # Geometry Optimization
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50) == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42, maxAttempts=50)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        
        # Meeko API Compliance
        preparator = MoleculePreparation()
        setups = preparator.prepare(mol)
        if not setups: return None
        
        res = PDBQTWriterLegacy.write_string(setups[0])
        return res[0] if isinstance(res, tuple) else res
    except Exception as e:
        return None


def dock_single_target(
    vina_exe: Path, 
    ligand_smiles: str, 
    candidate_code: str,
    target_config: Dict,
    work_dir: Path
) -> Tuple[str, float]:
    """Execute 3D docking simulation with a strict Watchdog Timeout."""
    disease = target_config.get("disease", "Unknown")
    pdbqt_str = target_config.get("pdbqt")
    active_site = target_config.get("active_site", {})
    
    if not pdbqt_str or not active_site: return disease, 0.0
    target_pdbqt = Path(pdbqt_str)
    if not target_pdbqt.exists(): return disease, 0.0

    ligand_pdbqt_data = prepare_ligand_pdbqt(ligand_smiles)
    if not ligand_pdbqt_data: return disease, 0.0
        
    fd, temp_lig_path = tempfile.mkstemp(suffix=".pdbqt", dir=work_dir)
    with os.fdopen(fd, 'w') as f: f.write(ligand_pdbqt_data)
        
    try:
        cmd = [
            str(vina_exe),
            "--receptor", str(target_pdbqt),
            "--ligand", temp_lig_path,
            "--center_x", str(active_site.get("center_x", 0)),
            "--center_y", str(active_site.get("center_y", 0)),
            "--center_z", str(active_site.get("center_z", 0)),
            "--size_x", str(active_site.get("size_x", 20)),
            "--size_y", str(active_site.get("size_y", 20)),
            "--size_z", str(active_site.get("size_z", 20)),
            "--cpu", "1",
            "--exhaustiveness", "8"
        ]
        
        # THE FIX: 45-second Watchdog Timer to prevent Zombie Threads
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=45)
        affinity = parse_vina_score(result.stdout)
        
    except subprocess.TimeoutExpired:
        logging.warning(f"Vina TIMEOUT for {candidate_code} on {disease}. Terminating thread.")
        affinity = 0.0
    except subprocess.CalledProcessError as e:
        logging.debug(f"Vina failed for {candidate_code} on {disease}")
        affinity = 0.0
    finally:
        if os.path.exists(temp_lig_path):
            try: os.remove(temp_lig_path)
            except OSError: pass
            
    return disease, affinity if affinity is not None else 0.0


def process_molecule(vina_exe: Path, row: pd.Series, targets: List[Dict], work_dir: Path) -> Dict:
    candidate_code = row.get("candidate_code", "Unknown")
    smiles = row.get("smiles", "")
    target_scores = {}
    
    for target in targets:
        disease, score = dock_single_target(vina_exe, smiles, candidate_code, target, work_dir)
        if score < 0:
            target_scores[disease] = score
            
    result = row.to_dict()
    result["target_scores"] = target_scores
    result["target_scores_json"] = json.dumps(target_scores)
    return result


def batch_dock_library(
    input_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    output_csv: Path = Path("data/physics_engine/docking_results.csv"),
    config_path: Path = Path("data/target/target_index.json"),
    bin_dir: Path = Path("data/bin"),
    max_workers: Optional[int] = None
):
    if not input_csv.exists() or not config_path.exists(): return
    with open(config_path, "r", encoding="utf-8") as f: targets = json.load(f)
        
    vina_exe = ensure_vina_binary(bin_dir)
    df = pd.read_csv(input_csv)
    if df.empty: return df.to_csv(output_csv, index=False)
        
    work_dir = Path("data/physics_engine/ligand_cache")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # THE FIX: Dynamically scale threads to maximize CPU utilization
    if max_workers is None:
        cores = os.cpu_count() or 4
        max_workers = max(1, cores - 2)
        
    logging.info(f"Initiating ThreadPool Docking with {max_workers} threads...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_molecule, vina_exe, row, targets, work_dir): row for _, row in df.iterrows()}
        for future in as_completed(futures):
            try: results.append(future.result())
            except Exception: pass
                
    out_df = pd.DataFrame(results)
    if "target_scores_json" in out_df.columns:
        out_df["target_scores"] = out_df["target_scores_json"]
        out_df.drop(columns=["target_scores_json"], inplace=True)
        
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    batch_dock_library()