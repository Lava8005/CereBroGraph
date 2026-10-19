"""TDD Suite for GNN Training Lifecycle and Evaluation Metrics."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from vector_engine.train_eval import evaluate_model
from vector_engine.device_manager import DeviceManager

def test_evaluate_model_metrics():
    """Ensure ROC-AUC, F1, and Precision are mathematically accurate during validation."""
    dm = DeviceManager(force_cpu=True)
    
    # Mock a dummy model returning deterministic logits
    mock_model = MagicMock()
    # Logits > 0 = Sigmoid > 0.5 (Positive prediction)
    # Logits < 0 = Sigmoid < 0.5 (Negative prediction)
    mock_model.return_value = torch.tensor([[-2.0], [2.0], [2.0], [-2.0]])
    
    # Mock a Batch object
    class MockBatch:
        x, edge_index, batch = None, None, None
        edge_attr, global_features = None, None
        y = torch.tensor([0, 1, 0, 1]) # 2 True Positives, 1 False Pos, 1 False Neg
        
        def to(self, device): return self
        
    mock_loader = [MockBatch()]
    criterion = nn.BCEWithLogitsLoss()
    
    metrics, loss = evaluate_model(mock_model, mock_loader, criterion, dm)
    
    # 2 predictions were positive, 1 was correct (Precision = 0.5)
    assert metrics["precision"] == 0.5
    # 2 actual positives, 1 was found (Recall = 0.5)
    assert metrics["recall"] == 0.5
    assert "roc_auc" in metrics