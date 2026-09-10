import csv
import logging
import os
import concurrent.futures
from itertools import islice
from pathlib import Path
from typing import Optional, Dict

from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def process_single_smiles(row: Dict[str, str]) -> Optional[Dict[str, str]]:
    raw_smiles = row.get("smiles", "")
    try:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None: 
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags: 
            return None
        largest_frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        clean_smiles = Chem.MolToSmiles(largest_frag, canonical=True)
        return {"source_id": row.get("source_id", row.get("zinc_id", "UNKNOWN")), "smiles": clean_smiles}
    except Exception:
        return None

def process_large_library(
    input_csv: Path = Path("data/library/raw/zinc_subset.csv"),
    output_csv: Path = Path("data/library/curated_library.csv"), 
    batch_size: int = 5000, 
    max_workers: Optional[int] = None
) -> int:
    workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    
    if not input_csv.exists():
        logging.warning(f"Missing raw library at {input_csv}. Creating mock candidate.")
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        input_csv.write_text("smiles,source_id\nCC(=O)OC1=CC=CC=C1C(=O)O,ASPIRIN\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["candidate_code", "source_id", "smiles"])
        writer.writeheader()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            while True:
                batch_rows = list(islice(reader, batch_size))
                if not batch_rows:
                    break
                
                results = list(executor.map(process_single_smiles, batch_rows))
                
                for res in results:
                    if res is not None:
                        valid_count += 1
                        res["candidate_code"] = f"OV-{valid_count:06d}"
                        writer.writerow(res)
                        
    logging.info(f"Curated {valid_count} molecules.")
    return valid_count

# Backward compatibility alias
process_library = process_large_library

if __name__ == "__main__":
    process_large_library()