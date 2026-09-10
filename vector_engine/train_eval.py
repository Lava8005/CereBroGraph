"""Model Training & Evaluation for Blood-Brain Barrier SE-GNN.

Orchestrates the PyTorch Geometric training loops, handles PCI-e CPU-to-GPU 
tensor transfers, and persists high-performing model checkpoints.
"""

import logging
import sys
from pathlib import Path

# Dynamically anchor Python's module search path to the project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def train_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """Execute one complete training epoch over the dataset."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        # THE FIX: JIT transfer of CPU batch to GPU VRAM
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass on GPU
        predictions = model(batch.x, batch.edge_index, batch.batch)
        
        # Calculate loss (Ensure dimensions match: [batch_size] vs [batch_size])
        loss = criterion(predictions.squeeze(), batch.y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Accumulate loss using .item() to prevent VRAM leaks
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> tuple[float, float]:
    """Evaluate model performance on a validation/test split and compute ROC-AUC."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # THE FIX: JIT transfer for evaluation batches
            batch = batch.to(device)
            
            predictions = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(predictions.squeeze(), batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            # Transfer predictions back to CPU for Scikit-Learn metrics
            all_preds.extend(predictions.squeeze().cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        logging.warning("ROC-AUC undefined: Only one class present in evaluation batch. Returning 0.5.")
        roc_auc = 0.5

    return avg_loss, roc_auc


def run_training_pipeline(epochs: int = 20, batch_size: int = 128):
    """Execute the full training loop and serialize the best model weights."""
    checkpoint_dir = Path(project_root) / "vector_engine" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_bbb_gnn.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initializing training on device: {device}")

    from vector_engine.gnn_model import BBBGraphModel
    model = BBBGraphModel(num_node_features=5, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Binary Cross Entropy for probability prediction
    criterion = nn.BCELoss() 

    try:
        from vector_engine.b3db_to_graph import get_b3db_dataloaders
        train_loader, val_loader = get_b3db_dataloaders(batch_size=batch_size)
    except ImportError as e:
        logging.error(f"Missing dataset loader: {e}")
        return

    best_roc_auc = 0.0
    
    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Pass device to the epoch functions
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_roc_auc = eval_epoch(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
        
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            logging.info(f"🌟 New best ROC-AUC ({best_roc_auc:.4f})! Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    logging.info(f"Training complete. Best weights saved at: {checkpoint_path}")

if __name__ == "__main__":
    run_training_pipeline(epochs=20)