"""Drug Likeness & Atom Whitelist Filter.

Evaluates synthetic accessibility, Lipinski violations, and strips molecules
containing metalloids/metals unsupported by AutoDock Vina to prevent downstream physics crashes.
"""

import csv
import logging
from itertools import islice
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# THE FIX: AutoDock Vina strictly supports only these atomic numbers
VINA_ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53} # H, C, N, O, F, P, S, Cl, Br, I

def is_vina_compatible(mol: Chem.Mol) -> bool:
    """Ensure the molecule only contains atoms supported by AutoDock Vina."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in VINA_ALLOWED_ATOMS:
            return False
    return True

def calculate_lipinski(mol: Chem.Mol) -> int:
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if Descriptors.NumHDonors(mol) > 5: violations += 1
    if Descriptors.NumHAcceptors(mol) > 10: violations += 1
    return violations

def calculate_cns_likeness(mol: Chem.Mol) -> bool:
    """Strict Blood-Brain Barrier (BBB) physicochemical heuristic filter."""
    try:
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        
        # Golden rules for CNS penetration (Pajouhesh & Lenz, 2005)
        if mw > 500 or tpsa > 90 or hbd > 3:
            return False
        return True
    except Exception:
        return False


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
                    if not is_vina_compatible(mol):
                        continue
                    
                    # THE FIX: Apply strict CNS limits
                    if not calculate_cns_likeness(mol):
                        continue
                        
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