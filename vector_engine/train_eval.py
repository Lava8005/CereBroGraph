"""Blood-Brain Barrier GNN Training Orchestrator.

Manages the lifecycle of the GINE Graph Neural Network. Implements
deterministic Bemis-Murcko scaffold splitting, Automatic Mixed Precision (AMP) 
Tensor Core acceleration, learning rate scheduling, and early stopping guards.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vector_engine.b3db_loader import load_b3db_dataset
from vector_engine.gnn_model import BBBGraphModel
from vector_engine.device_manager import default_device_manager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, dm) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = dm.to_device(batch)
            
            with dm.autocast():
                logits = model(
                    batch.x, 
                    batch.edge_index, 
                    batch.batch, 
                    edge_attr=batch.edge_attr, 
                    global_features=batch.global_features
                )
                loss = criterion(logits.view(-1), batch.y.float())
                
            total_loss += loss.item()
            probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc_auc = 0.5
        
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    
    try:
        pr_auc = average_precision_score(all_labels, all_preds)
    except ValueError:
        pr_auc = float("nan")

    metrics = {
        "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "precision": precision, "recall": recall,
        "accuracy": accuracy_score(all_labels, preds_binary),
        "balanced_accuracy": balanced_accuracy_score(all_labels, preds_binary),
        "mcc": matthews_corrcoef(all_labels, preds_binary),
    }
    return metrics, avg_loss


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training (initialisation, shuffling, dropout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 7,
    seed: int = 42,
    split_mode: str = "scaffold",
    checkpoint_path: Optional[Path] = None,
    metrics_path: Optional[Path] = Path("data/vector_engine/bbb_eval_metrics.json"),
    datasets: Optional[tuple] = None,
) -> Optional[dict]:
    """Train the BBB GINE model and return validation and test metrics.

    `checkpoint_path` defaults to the shipped checkpoint location; pass another path
    (for example in experiments) to avoid overwriting it. `datasets` may supply a
    pre-built (train, val, test) tuple.
    """
    dm = default_device_manager
    set_seed(seed)
    logging.info(f"Initializing GINE training on device: {dm.device.type} (seed={seed}, split={split_mode})")

    if datasets is None:
        datasets = load_b3db_dataset(split_mode=split_mode, seed=seed)
    train_dataset, val_dataset, test_dataset = datasets
    if not train_dataset or not val_dataset or not test_dataset:
        logging.error("Dataset loading failed. Aborting training.")
        return None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
    model = dm.to_device(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler('cuda') if dm.is_cuda else None
    
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_metrics = {}
    
    best_model_path = Path(checkpoint_path) if checkpoint_path else Path("vector_engine/checkpoints/best_bbb_gnn.pt")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_run = 0
    
    logging.info(f"Starting training for max {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = dm.to_device(batch)
            optimizer.zero_grad()
            
            with dm.autocast():
                logits = model(
                    batch.x, 
                    batch.edge_index, 
                    batch.batch, 
                    edge_attr=batch.edge_attr, 
                    global_features=batch.global_features
                )
                loss = criterion(logits.view(-1), batch.y.float())
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
        val_metrics, val_loss = evaluate_model(model, val_loader, criterion, dm)
        val_auc = val_metrics["roc_auc"]
        
        logging.info(f"Epoch {epoch:03d} | Train Loss: {total_loss/len(train_loader):.4f} | "
                     f"Val Loss: {val_loss:.4f} | ROC-AUC: {val_auc:.4f} | F1: {val_metrics['f1']:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.warning(f"Early stopping at epoch {epoch}. Reverting to best weights.")
                break
                
        dm.empty_cache()
                
    logging.info("Training complete. Loading best weights for Test Set evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=dm.device, weights_only=True))
    test_metrics, test_loss = evaluate_model(model, test_loader, criterion, dm)
    
    logging.info(f"Held-Out TEST SET | Loss: {test_loss:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | F1: {test_metrics['f1']:.4f}")
    
    result = {
        "seed": seed,
        "split_mode": split_mode,
        "epochs_run": epochs_run,
        "sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        "val": best_metrics,
        "test": test_metrics,
    }
    # Keys kept for backwards compatibility with earlier metric files
    result["val"]["test_roc_auc"] = test_metrics["roc_auc"]
    result["val"]["test_f1"] = test_metrics["f1"]

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
    return result

if __name__ == "__main__":
    run_training()