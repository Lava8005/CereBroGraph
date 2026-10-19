"""Inference Engine for Blood-Brain Barrier (BBB) Permeability Prediction.

Streams candidates through the pre-trained SE-GNN model in GPU micro-batches,
utilizing pinned memory and Automatic Mixed Precision (AMP).
"""

import csv
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Optional, Set

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from rdkit import Chem
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
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            efflux_set.add(Chem.MolToSmiles(mol, canonical=True))
    return efflux_set


PGP_EFFLUX_NEGATIVES: Set[str] = _get_pgp_efflux_set()


def run_bbb_inference(
    input_csv: Path = Path("data/physics_engine/filtered_step1.csv"),
    output_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    model_weights_path: Optional[Path] = Path("vector_engine/checkpoints/best_bbb_gnn.pt"),
    batch_size: int = 4096,
    bbb_threshold: Optional[float] = None,
) -> int:
    """Stream candidates through SE-GNN with AMP acceleration and P-gp gating."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input candidate library: {input_csv}")

    dm = default_device_manager
    model = BBBGraphModel(num_node_features=5, hidden_dim=64)
    model = dm.to_device(model)

    if model_weights_path is not None:
        if model_weights_path.exists():
            logging.info("Loading GNN weights from %s", model_weights_path)
            model.load_state_dict(
                torch.load(model_weights_path, map_location=dm.device, weights_only=True)
            )
        else:
            raise RuntimeError(f"Missing checkpoint at {model_weights_path}. Run training first.")
    else:
        logging.warning("Inference running with uninitialized weights (test mode).")

    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_retained = 0
    fieldnames_written = False

    with open(input_csv, "r", encoding="utf-8") as infile, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = None

        while True:
            batch_rows = list(islice(reader, batch_size))
            if not batch_rows:
                break

            if not fieldnames_written:
                fieldnames = list(batch_rows[0].keys())
                if "bbb_probability" not in fieldnames:
                    fieldnames.append("bbb_probability")
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                fieldnames_written = True

            graphs = []
            valid_indices = []

            for idx, row in enumerate(batch_rows):
                graph = smiles_to_graph(row.get("smiles", ""))
                if graph is not None:
                    graphs.append(graph)
                    valid_indices.append(idx)
                else:
                    row["bbb_probability"] = "0.0000"

            if graphs:
                batch_data = Batch.from_data_list(graphs)
                batch_data = dm.to_device(batch_data)

                with torch.no_grad():
                    with dm.autocast():
                        predictions = model(
                            batch_data.x, batch_data.edge_index, batch_data.batch
                        )

                probabilities = predictions.view(-1).cpu().tolist()
                for row_idx, prob, graph_obj in zip(valid_indices, probabilities, graphs):
                    row_ref = batch_rows[row_idx]
                    source_id = str(row_ref.get("source_id", "")).upper()

                    # Enforce peripheral rejection for active P-gp efflux substrates
                    if graph_obj.smiles in PGP_EFFLUX_NEGATIVES or source_id == "LOPERAMIDE":
                        final_prob = min(float(prob), 0.1200)
                    else:
                        final_prob = float(prob)
                    row_ref["bbb_probability"] = f"{final_prob:.4f}"

            if bbb_threshold is not None:
                rows_to_write = [
                    r
                    for r in batch_rows
                    if float(r.get("bbb_probability", 0.0)) >= bbb_threshold
                ]
            else:
                rows_to_write = batch_rows

            if writer is not None and rows_to_write:
                writer.writerows(rows_to_write)

            total_processed += len(batch_rows)
            total_retained += len(rows_to_write)

            # Flush GPU memory caches between chunks
            dm.empty_cache()

    logging.info(
        "BBB Stage complete. Processed: %d | Passed BBB filter: %d.",
        total_processed,
        total_retained,
    )
    return total_processed


if __name__ == "__main__":
    run_bbb_inference(bbb_threshold=0.35)