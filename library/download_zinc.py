"""ZINC Database Network Streamer.

Reads standard ZINC fallback URIs, streams SMILES over HTTP without exhausting RAM, 
truncates exactly at the target count, and injects FDA validation controls.
"""

import csv
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# FDA Controls required for Scientific Validation (Scientifically Accurate Topologies)
FDA_CONTROLS = [
    {"smiles": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "source_id": "DONEPEZIL"},
    {"smiles": "CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O", "source_id": "SAFINAMIDE"},
    {"smiles": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", "source_id": "LOPERAMIDE"},
    {"smiles": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2", "source_id": "RISPERIDONE"},
    {"smiles": "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1", "source_id": "PIMAVANSERIN"}
]

def process_smi_line(line_str: str) -> Optional[Dict[str, str]]:
    parts = line_str.strip().split()
    if len(parts) >= 2:
        return {"smiles": parts[0], "source_id": parts[1]}
    return None

def stream_zinc_subset(
    uri_list_path: Path = Path("ZINC-downloader-2D-smi.uri"),
    output_csv: Path = Path("data/library/raw/zinc_subset.csv"),
    target_count: int = 50000
):
    urls = []
    if uri_list_path.exists():
        with open(uri_list_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip().startswith("http")]

    if not urls:
        logging.warning(
            f"URI list {uri_list_path} not found. Falling back to the MoleculeNet HIV.csv set "
            "(a public drug-like library, NOT ZINC). Download a ZINC '2D-smi' URI file from "
            "https://zinc.docking.org/tranches/home/ and pass it with --uri-file to use ZINC."
        )
        urls = ["https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["smiles", "source_id"])
        writer.writeheader()
        
        for control in FDA_CONTROLS:
            writer.writerow(control)
            collected += 1
            
        logging.info(f"Injected {len(FDA_CONTROLS)} FDA controls.")

        for url in urls:
            if collected >= target_count:
                break
                
            logging.info(f"Streaming from Tranche: {url}")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    for line_bytes in response:
                        if collected >= target_count:
                            break
                        
                        line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                        if not line_str or line_str.lower().startswith("smiles"):
                            continue
                            
                        # Handle CSV comma formats or space/tab formats
                        if "," in line_str:
                            parts = line_str.split(',')
                            parsed = {"smiles": parts[0].strip('"'), "source_id": f"MOL-{collected}"} if len(parts) >= 1 else None
                        else:
                            parsed = process_smi_line(line_str)
                            
                        if parsed:
                            writer.writerow(parsed)
                            collected += 1
                            
                        if collected % 5000 == 0:
                            logging.info(f"  ... Collected {collected} molecules")
                            
            except Exception as e:
                logging.warning(f"Failed to stream from {url}: {e}. Moving to next tranche...")
                continue

    if collected < len(FDA_CONTROLS) + 10:
        logging.error("Failed to collect meaningful subset. Network stream failed.")
        sys.exit(1)

    logging.info(f"Success! {collected} molecules saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000, help="Total molecules to collect (incl. controls).")
    parser.add_argument("--uri-file", type=Path, default=Path("ZINC-downloader-2D-smi.uri"),
                        help="ZINC downloader .uri file listing tranche URLs.")
    args = parser.parse_args()
    stream_zinc_subset(uri_list_path=args.uri_file, target_count=args.count)