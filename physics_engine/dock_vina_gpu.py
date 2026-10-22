import json
import logging
import subprocess
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def execute_gpu_batch(
    vina_gpu_exe: str, 
    ligand_dir: str, 
    target: dict, 
    output_dir: str, 
    timeout: int = 3600
) -> bool:
    """Executes GPU batch. CLI flags MUST be empirically verified against specific fork documentation."""
    grid_path = target.get("grid", "")
    with open(grid_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    # Note: Flag syntax varies between DeltaGroup Vina-GPU forks. Verify locally.
    cmd = [
        vina_gpu_exe,
        "--receptor", target.get("pdbqt", ""),
        "--ligand_directory", ligand_dir,
        "--output_directory", output_dir,
        "--center_x", str(g["center_x"]), "--center_y", str(g["center_y"]), "--center_z", str(g["center_z"]),
        "--size_x", str(g["size_x"]), "--size_y", str(g["size_y"]), "--size_z", str(g["size_z"]),
        "--thread", "8000"
    ]

    process = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logging.error(f"Execution failed. Exit code {process.returncode}")
            return False
        return True
        
    except subprocess.TimeoutExpired:
        logging.error(f"Batch exceeded timeout of {timeout}s.")
        return False
    except BaseException as e:
        # Catches KeyboardInterrupt, SystemExit (if handled), and standard Exceptions
        logging.error(f"Process interrupted/failed: {type(e).__name__}")
        raise
    finally:
        if process is not None and process.poll() is None:
            logging.warning("Terminating orphaned GPU process...")
            process.kill()
            process.communicate()

def parse_gpu_results(output_dir: str, expected_count: int) -> Dict[str, float]:
    """Strictly parses outputs. Fails loudly on data loss or malformed files."""
    results = {}
    out_files = list(Path(output_dir).glob("*_out.pdbqt"))
    
    if len(out_files) != expected_count:
        raise ValueError(f"Output mismatch: Expected {expected_count} results, found {len(out_files)}.")

    for out_file in out_files:
        cand_code = out_file.name.replace("_out.pdbqt", "")
        affinity = None
        
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    try:
                        affinity = float(line.split()[3])
                        break
                    except (IndexError, ValueError):
                        break
                        
        if affinity is None:
            raise ValueError(f"Missing or malformed VINA RESULT in {out_file.name}")
            
        results[cand_code] = affinity
        
    return results