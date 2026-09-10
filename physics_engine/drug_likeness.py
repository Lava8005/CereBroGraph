import csv
import logging
from itertools import islice
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def calculate_lipinski(mol: Chem.Mol) -> int:
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if Descriptors.NumHDonors(mol) > 5: violations += 1
    if Descriptors.NumHAcceptors(mol) > 10: violations += 1
    return violations

def batch_compute_drug_likeness(
    input_csv: Path = Path("data/library/curated_library.csv"),
    output_csv: Path = Path("data/physics_engine/filtered_step1.csv"),
    batch_size: int = 5000
) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing upstream library: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    survivors = 0
    
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for col in ["qed", "lipinski_violations"]:
            if col not in fieldnames:
                fieldnames.append(col)
                
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            batch_rows = list(islice(reader, batch_size))
            if not batch_rows:
                break
                
            valid_batch = []
            for row in batch_rows:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is not None:
                    violations = calculate_lipinski(mol)
                    if violations <= 1:
                        row["qed"] = round(QED.qed(mol), 4)
                        row["lipinski_violations"] = violations
                        valid_batch.append(row)
                        survivors += 1
                        
            if valid_batch:
                writer.writerows(valid_batch)
                
    logging.info(f"Drug likeness filter passed {survivors} molecules to {output_csv}")
    return survivors

if __name__ == "__main__":
    batch_compute_drug_likeness()