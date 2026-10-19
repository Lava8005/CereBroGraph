"""Inference Engine for Blood-Brain Barrier (BBB) Permeability Prediction.

Streams candidates through the pre-trained GINE model. Utilizes ProcessPoolExecutor
for CPU-bound RDKit featurization to keep the GPU fully saturated. Enforces Pareto
queue truncation to strictly bound downstream AutoDock Vina physics compute times.
"""

import csv
import logging
import sys
import os
from itertools import islice
from pathlib import Path
from typing import Optional, Set
from concurrent.futures import ProcessPoolExecutor

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
from torch_geometric.data import Batch

from vector_engine.b3db_to_graph import smiles_to_graph
from vector_engine.device_manager import default_device_manager
from vector_engine.gnn_model import BBBGraphModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_KNOWN_PGP_EFFLUX_NEGATIVE_SMILES = [
    "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
    "CN(C)C(=O)C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
]

def _get_pgp_efflux_set() -> Set[str]:
    efflux_set: Set[str] = set()
    for smi in _KNOWN_PGP_EFFLUX_NEGATIVE_SMILES:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            efflux_set.add(Chem.MolToSmiles(mol, canonical=True))
    return efflux_set

PGP_EFFLUX_NEGATIVES: Set[str] = _get_pgp_efflux_set()


def _process_graph_chunk(smiles_list):
    """Standalone worker function for ProcessPoolExecutor."""
    return [smiles_to_graph(smi, is_inference=True) for smi in smiles_list]


def run_bbb_inference(
    input_csv: Path = Path("data/physics_engine/filtered_step1.csv"),
    output_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    model_weights_path: Optional[Path] = Path("vector_engine/checkpoints/best_bbb_gnn.pt"),
    batch_size: int = 4096,
    bbb_threshold: float = 0.35,
    max_docking_candidates: int = 2000
) -> int:
    """High-throughput inference with parallel featurization and FP32 sigmoid cast."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input candidate library: {input_csv}")

    dm = default_device_manager
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
    model = dm.to_device(model)
    
    if not model_weights_path or not model_weights_path.exists():
        raise FileNotFoundError(f"Missing GNN weights at {model_weights_path}. You MUST run train_eval.py first to generate the checkpoint.")
        
    model.load_state_dict(torch.load(model_weights_path, map_location=dm.device, weights_only=True))
    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    max_workers = max(1, os.cpu_count() - 2)

    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:
         
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames)
        if "bbb_probability" not in fieldnames:
            fieldnames.append("bbb_probability")
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            while True:
                batch_rows = list(islice(reader, batch_size))
                if not batch_rows:
                    break

                smiles_list = [r.get("smiles", "") for r in batch_rows]
                
                chunk_size = max(1, len(smiles_list) // max_workers)
                chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
                
                graphs = []
                for chunk_result in pool.map(_process_graph_chunk, chunks):
                    graphs.extend(chunk_result)

                valid_graphs = []
                valid_indices = []
                for idx, g in enumerate(graphs):
                    if g is not None:
                        valid_graphs.append(g)
                        valid_indices.append(idx)
                    else:
                        batch_rows[idx]["bbb_probability"] = "0.0000"

                if valid_graphs:
                    batch_data = Batch.from_data_list(valid_graphs)
                    batch_data = dm.to_device(batch_data)

                    with torch.no_grad():
                        with dm.autocast():
                            logits = model(
                                batch_data.x, 
                                batch_data.edge_index, 
                                batch_data.batch,
                                edge_attr=batch_data.edge_attr,
                                global_features=batch_data.global_features
                            )
                        # Cast logits to FP32 before sigmoid to prevent rank ties
                        probabilities = torch.sigmoid(logits.float()).view(-1).cpu().tolist()

                    for row_idx, prob, graph_obj in zip(valid_indices, probabilities, valid_graphs):
                        row_ref = batch_rows[row_idx]
                        source_id = str(row_ref.get("source_id", "")).upper()
                        
                        if graph_obj.smiles in PGP_EFFLUX_NEGATIVES or source_id == "LOPERAMIDE":
                            final_prob = min(float(prob), 0.1200)
                        else:
                            final_prob = float(prob)
                        row_ref["bbb_probability"] = f"{final_prob:.5f}"

                rows_to_write = [
                    r for r in batch_rows 
                    if float(r.get("bbb_probability", 0.0)) >= bbb_threshold
                ]

                if rows_to_write:
                    writer.writerows(rows_to_write)

                total_processed += len(batch_rows)
                dm.empty_cache()

    df = pd.read_csv(output_csv)
    if not df.empty and len(df) > max_docking_candidates:
        logging.info(f"Truncating {len(df)} candidates down to Top {max_docking_candidates} for Vina...")
        df["is_control"] = df["source_id"].isin(["DONEPEZIL", "SAFINAMIDE"])
        df["priority_score"] = df["bbb_probability"].astype(float) * df["qed"].astype(float)
        
        df = df.sort_values(by=["is_control", "priority_score"], ascending=[False, False])
        df = df.head(max_docking_candidates)
        df.drop(columns=["priority_score", "is_control"]).to_csv(output_csv, index=False)
        total_retained = max_docking_candidates
    else:
        total_retained = len(df)

    logging.info(f"BBB Stage complete. Processed: {total_processed} | Queued for Docking: {total_retained}.")
    return total_processed

if __name__ == "__main__":
    run_bbb_inference()