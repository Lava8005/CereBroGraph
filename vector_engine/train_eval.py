"""Blood-Brain Barrier GNN Training Orchestrator.

Manages the lifecycle of the GINE Graph Neural Network. Implements
deterministic Bemis-Murcko scaffold splitting, Automatic Mixed Precision (AMP) 
Tensor Core acceleration, learning rate scheduling, and early stopping guards.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure absolute imports resolve correctly from the command line
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vector_engine.b3db_loader import load_b3db_dataset
from vector_engine.gnn_model import BBBGraphModel
from vector_engine.device_manager import default_device_manager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, dm) -> tuple:
    """Run validation utilizing high-precision FP32 metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = dm.to_device(batch)
            
            # Forward pass under AMP
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
            
            # THE FIX: High-Precision Sigmoid for strict mathematical ranking
            probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    # Binarize for classification metrics
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc_auc = 0.5 # Failsafe for homogeneous batches
        
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    
    metrics = {
        "roc_auc": roc_auc, 
        "f1": f1, 
        "precision": precision, 
        "recall": recall
    }
    
    return metrics, avg_loss


def run_training(epochs: int = 50, batch_size: int = 256, patience: int = 7, seed: int = 42):
    """Execute the full GINE training loop with anti-overfitting guards."""
    dm = default_device_manager
    logging.info(f"Initializing GINE training on device: {dm.device.type} (seed={seed})")
    
    # 1. Dataset Injection (Deterministic Split)
    train_dataset, val_dataset = load_b3db_dataset(split_mode="scaffold", seed=seed)
    
    if not train_dataset or not val_dataset:
        logging.error("Dataset loading failed. Aborting training.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Upgraded Architecture Instantiation
    model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
    model = dm.to_device(model)
    
    # 3. Optimizers & Training Guards
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_metrics = {}
    
    checkpoints_dir = Path("vector_engine/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "best_bbb_gnn.pt"
    
    logging.info(f"Starting training for max {epochs} epochs on Bemis-Murcko scaffold split...")
    
    for epoch in range(1, epochs + 1):
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
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Guard against exploding gradients
            optimizer.step()
            
            total_loss += loss.item()
            
        # 4. Evaluation & Scheduling
        val_metrics, val_loss = evaluate_model(model, val_loader, criterion, dm)
        val_auc = val_metrics["roc_auc"]
        
        logging.info(f"Epoch {epoch:03d} | Train Loss: {total_loss/len(train_loader):.4f} | "
                     f"Val Loss: {val_loss:.4f} | ROC-AUC: {val_auc:.4f} | "
                     f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        scheduler.step(val_auc)
        
        # 5. Early Stopping Guard
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info("  -> New optimal weights saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.warning(f"Early stopping triggered at epoch {epoch}! Reverting to optimal weights.")
                break
                
        dm.empty_cache()
                
    logging.info(f"Training complete. Best Val ROC-AUC: {best_val_auc:.4f}.")
    
    # Write payload for analysis scripts
    metrics_path = Path("data/vector_engine/bbb_eval_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)


if __name__ == "__main__":
    run_training()