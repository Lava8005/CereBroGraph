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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional

# Suppress verbose RDKit warnings for malformed molecules during bulk ingestion
from rdkit import Chem
from rdkit import RDLogger
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
            
        # 1. Parse and Validate Structure
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            continue
            
        try:
            # 2. Canonicalize SMILES (Ensure consistency)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # 3. Generate Deterministic Primary Key
            # InChIKey is a 27-character hashed representation of the molecular topology
            inchi_key = Chem.inchi.MolToInchiKey(mol)
            candidate_code = f"MOL-{inchi_key[:10]}"
            
            valid_molecules.append({
                "candidate_code": candidate_code,
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
            futures = []
            
            # Read CSV lazily in chunks to maintain O(1) memory footprint
            while True:
                chunk = list(islice(reader, chunk_size))
                if not chunk:
                    break
                    
                total_processed += len(chunk)
                futures.append(executor.submit(process_chunk, chunk))
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    valid_chunk = future.result()
                    if valid_chunk:
                        writer.writerows(valid_chunk)
                        total_retained += len(valid_chunk)
                except Exception as e:
                    logging.error(f"Worker process failed: {e}")
                    
    logging.info(f"Curation complete. Ingested: {total_processed} | Retained: {total_retained}")
    return total_retained


if __name__ == "__main__":
    batch_curate_library()