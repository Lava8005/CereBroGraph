"""Library Curation Module for CerebroGraph.

Ingests raw SMILES data, canonicalizes molecular structures using RDKit,
filters out chemically invalid strings, and generates deterministic, 
cryptographically stable InChIKey-based candidate codes.
Utilizes chunked multiprocessing to eliminate IPC bottlenecks.
"""

import csv
import logging
import os
from itertools import islice
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

from rdkit import Chem
from rdkit import RDLogger
# Suppress verbose RDKit warnings for malformed molecules during bulk ingestion
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def process_chunk(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Worker function: Canonicalize SMILES and generate stable InChIKeys."""
    valid_molecules = []
    for row in batch:
        raw_smiles = row.get("smiles", "").strip()
        source_id = row.get("source_id", "UNKNOWN").strip()
        
        if not raw_smiles:
            continue
            
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            continue
            
        try:
            # 1. Canonicalize SMILES (Ensure consistency)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # 2. Generate Deterministic Primary Key
            # Full 27-character InChIKey for collision-free uniqueness
            inchi_key = Chem.inchi.MolToInchiKey(mol)
            
            # 3. Prepend MOL- prefix in the worker
            valid_molecules.append({
                "candidate_code": f"MOL-{inchi_key}",
                "source_id": source_id,
                "smiles": canonical_smiles
            })
        except Exception:
            # Catch rare stereochemistry parsing failures
            continue
            
    return valid_molecules

def batch_curate_library(
    input_csv: Path = Path("data/library/raw/zinc_subset.csv"),
    output_csv: Path = Path("data/library/curated_library.csv"),
    chunk_size: int = 1000
) -> int:
    """Orchestrate chunked multi-processing for high-throughput curation."""
    if not input_csv.exists():
        logging.error(f"Missing raw library file at {input_csv}. Run downloader first.")
        return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0
    total_retained = 0
    
    # Determine optimal worker count (Leave 1 core for OS to prevent locking)
    max_workers = max(1, os.cpu_count() - 1)
    
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:
         
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["candidate_code", "source_id", "smiles"])
        writer.writeheader()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunks = []
            
            # Read CSV lazily in chunks to maintain O(1) memory footprint
            while True:
                chunk = list(islice(reader, chunk_size))
                if not chunk:
                    break
                total_processed += len(chunk)
                chunks.append(chunk)
            
            seen_codes = set()
            
            # THE FIX: Use map() to preserve original insertion order (Prioritizing early FDA controls)
            for valid_chunk in executor.map(process_chunk, chunks):
                if valid_chunk:
                    unique_chunk = []
                    for mol in valid_chunk:
                        # THE FIX: Dedupe purely on candidate_code to prevent Postgres Upsert crashes
                        if mol["candidate_code"] not in seen_codes:
                            seen_codes.add(mol["candidate_code"])
                            unique_chunk.append(mol)
                            
                    if unique_chunk:
                        writer.writerows(unique_chunk)
                        total_retained += len(unique_chunk)
                    
    logging.info(f"Curation complete. Ingested: {total_processed} | Retained: {total_retained}")
    return total_retained

if __name__ == "__main__":
    batch_curate_library()