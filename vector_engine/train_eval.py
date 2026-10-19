"""Model Training & Multi-Metric Evaluation for Blood-Brain Barrier SE-GNN.

Orchestrates PyTorch Geometric training loops on the Bemis-Murcko scaffold split,
handles PCI-e CPU-to-GPU tensor transfers, logs comprehensive classification
metrics (ROC-AUC, Precision, Recall, F1, Accuracy), and persists checkpoints.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Phase 2 Task 8 Numerical Validation Threshold
TARGET_ROC_AUC_THRESHOLD = 0.80
SPLIT_SEED = 42


def compute_classification_metrics(
    y_true: List[float],
    y_probs: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute ROC-AUC, Precision, Recall, F1-score, and Accuracy."""
    if not y_true or not y_probs:
        return {
            "roc_auc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    y_bin = [1 if p >= threshold else 0 for p in y_probs]
    y_true_int = [int(round(y)) for y in y_true]

    try:
        if len(set(y_true_int)) > 1:
            roc_auc = float(roc_auc_score(y_true_int, y_probs))
        else:
            logging.warning("ROC-AUC undefined: Only one class present. Returning 0.5.")
            roc_auc = 0.5
    except ValueError:
        roc_auc = 0.5

    precision = float(precision_score(y_true_int, y_bin, zero_division=0))
    recall = float(recall_score(y_true_int, y_bin, zero_division=0))
    f1 = float(f1_score(y_true_int, y_bin, zero_division=0))
    accuracy = float(accuracy_score(y_true_int, y_bin))

    return {
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one complete training epoch over the dataset."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(predictions.view(-1), batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / max(1, len(loader.dataset))


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_full_metrics: bool = False,
):
    """Evaluate model performance on a held-out split."""
    model.eval()
    total_loss = 0.0
    all_preds: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index, batch.batch)
            preds_flat = predictions.view(-1)
            labels_flat = batch.y.view(-1)

            loss = criterion(preds_flat, labels_flat)
            total_loss += loss.item() * batch.num_graphs

            all_preds.extend(preds_flat.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = compute_classification_metrics(all_labels, all_preds)

    if return_full_metrics:
        return avg_loss, metrics
    return avg_loss, metrics["roc_auc"]


def run_training_pipeline(
    epochs: int = 30,
    batch_size: int = 128,
    random_seed: int = SPLIT_SEED,
) -> Dict[str, float]:
    """Execute the full scaffold-split training loop and serialize best weights & metrics."""
    torch.manual_seed(random_seed)

    checkpoint_dir = Path(project_root) / "vector_engine" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_bbb_gnn.pt"

    metrics_path = Path(project_root) / "data" / "vector_engine" / "bbb_eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initializing SE-GNN training on device: {device} (seed={random_seed})")

    from vector_engine.gnn_model import BBBGraphModel
    from vector_engine.b3db_to_graph import get_b3db_dataloaders

    model = BBBGraphModel(num_node_features=5, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()

    train_loader, val_loader = get_b3db_dataloaders(
        batch_size=batch_size,
        val_split=0.2,
        random_seed=random_seed,
        split_mode="scaffold",
        exclude_controls=True,
    )

    best_roc_auc = 0.0
    best_metrics: Dict[str, float] = {}

    logging.info(f"Starting training for {epochs} epochs on Bemis-Murcko scaffold split...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = eval_epoch(
            model, val_loader, criterion, device, return_full_metrics=True
        )
        val_roc_auc = metrics["roc_auc"]

        logging.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"ROC-AUC: {val_roc_auc:.4f} | Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
        )

        if val_roc_auc >= best_roc_auc:
            best_roc_auc = val_roc_auc
            best_metrics = {
                "split_mode": "bemis_murcko_scaffold",
                "split_seed": random_seed,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "best_epoch": epoch,
                "val_loss": round(val_loss, 4),
                **metrics,
                "passes_auc_criterion": bool(val_roc_auc >= TARGET_ROC_AUC_THRESHOLD),
            }
            torch.save(model.state_dict(), checkpoint_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)

    logging.info(
        f"Training complete. Best Val ROC-AUC: {best_roc_auc:.4f} "
        f"(Threshold >= {TARGET_ROC_AUC_THRESHOLD}: {best_metrics.get('passes_auc_criterion')}). "
        f"Saved checkpoint to {checkpoint_path} and metrics to {metrics_path}."
    )
    return best_metrics


if __name__ == "__main__":
    run_training_pipeline(epochs=30)