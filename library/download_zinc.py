"""ZINC Database Network Streamer.

Reads the ZINC URI list, streams SMILES over HTTP without exhausting RAM, 
truncates exactly at the target count, and injects FDA validation controls.
"""

import csv
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# FDA Controls required for Scientific Validation 2
FDA_CONTROLS = [
    {"smiles": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "source_id": "DONEPEZIL"},
    {"smiles": "CC(Nc1ccc(OCc2ccccc2F)cc1)C(N)=O", "source_id": "SAFINAMIDE"},
    {"smiles": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", "source_id": "LOPERAMIDE"},
    {"smiles": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2", "source_id": "RISPERIDONE"},
    {"smiles": "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1", "source_id": "PIMAVANSERIN"}
]

def process_smi_line(line_str: str) -> Optional[Dict[str, str]]:
    """Parse a space or tab separated ZINC SMILES line."""
    parts = line_str.strip().split()
    if len(parts) >= 2:
        return {"smiles": parts[0], "source_id": parts[1]}
    return None

def stream_zinc_subset(
    uri_list_path: Path = Path("ZINC-downloader-2D-smi.uri"),
    output_csv: Path = Path("data/library/raw/zinc_subset.csv"),
    target_count: int = 50000
):
    if not uri_list_path.exists():
        logging.error(f"Missing ZINC URI list at {uri_list_path}. Please ensure it is in the root directory.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(uri_list_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]

    if not urls:
        logging.error("No valid URLs found in the URI list.")
        return

    collected = 0
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["smiles", "source_id"])
        writer.writeheader()
        
        # 1. Inject FDA Controls First so they are processed by the pipeline
        for control in FDA_CONTROLS:
            writer.writerow(control)
            collected += 1
            
        logging.info(f"Injected {len(FDA_CONTROLS)} FDA controls for Validation 2.")

        # 2. Stream from ZINC until target_count is reached
        for url in urls:
            if collected >= target_count:
                break
                
            logging.info(f"Streaming from ZINC Tranche: {url}")
            try:
                # Use urllib to open stream with a standard User-Agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, timeout=20) as response:
                    for line_bytes in response:
                        if collected >= target_count:
                            break
                        
                        line_str = line_bytes.decode('utf-8', errors='ignore')
                        if line_str.lower().startswith("smiles"):  # Skip headers
                            continue
                            
                        parsed = process_smi_line(line_str)
                        if parsed:
                            writer.writerow(parsed)
                            collected += 1
                            
                        if collected % 10000 == 0:
                            logging.info(f"  ... Collected {collected} molecules")
                            
            except Exception as e:
                logging.warning(f"Failed to stream from {url}: {e}. Moving to next tranche...")
                continue

    logging.info(f"Success! {collected} molecules saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZINC Library Network Streamer.")
    parser.add_argument("--count", type=int, default=50000, 
                        help="Exact number of molecules to extract. Default is 50,000.")
    args = parser.parse_args()
    
    stream_zinc_subset(target_count=args.count)