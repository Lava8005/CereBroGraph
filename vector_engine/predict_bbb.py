"""Inference engine for Blood-Brain Barrier (BBB) permeability prediction.

Streams candidates through an SE-GNN model in GPU/CPU micro-batches.
Strictly requires a pre-trained model checkpoint to execute.
"""

import csv
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Optional

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch_geometric.data import Batch

from vector_engine.b3db_to_graph import smiles_to_graph
from vector_engine.gnn_model import BBBGraphModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_compute_device() -> torch.device:
    """Select compute hardware based on CUDA availability."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logging.info("CUDA detected: %s. Using GPU acceleration.", device_name)
        return torch.device("cuda:0")
    logging.info("CUDA unavailable. Running inference on CPU fallback.")
    return torch.device("cpu")


def run_bbb_inference(
    input_csv: Path = Path("data/physics_engine/filtered_step1.csv"),
    output_csv: Path = Path("data/physics_engine/filtered_step2.csv"),
    model_weights_path: Optional[Path] = Path("vector_engine/checkpoints/best_bbb_gnn.pt"),
    batch_size: int = 4096,
) -> int:
    """Stream inputs through GNN model, enforcing strict pre-trained weight requirements."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input candidate library: {input_csv}")

    device = get_compute_device()
    model = BBBGraphModel(num_node_features=5, hidden_dim=64).to(device)

    if model_weights_path and model_weights_path.exists():
        logging.info("Loading GNN weights from %s", model_weights_path)
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    else:
        logging.error("CRITICAL: Untrained GNN detected. Random weights will invalidate screening.")
        raise RuntimeError(
            f"Missing model checkpoint at {model_weights_path}. "
            "You MUST run 'python vector_engine/train_eval.py' to generate valid weights before screening."
        )

    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    fieldnames_written = False

    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

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
                    row["bbb_probability"] = "0.5000"

            if graphs:
                batch_data = Batch.from_data_list(graphs).to(device)
                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            predictions = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    else:
                        predictions = model(batch_data.x, batch_data.edge_index, batch_data.batch)

                probabilities = predictions.view(-1).cpu().tolist()
                for row_idx, prob in zip(valid_indices, probabilities):
                    batch_rows[row_idx]["bbb_probability"] = f"{float(prob):.4f}"

            if writer is not None:
                writer.writerows(batch_rows)
            total_processed += len(batch_rows)

    logging.info("BBB inference complete. Total records: %d.", total_processed)
    return total_processed


if __name__ == "__main__":
    run_bbb_inference()