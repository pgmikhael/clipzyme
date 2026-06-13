from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import macro_f1_from_predictions


@dataclass
class MLPConfig:
    name: str
    hidden_dims: List[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    loss_type: str  # "ce" or "focal"
    use_class_weights: bool
    focal_gamma: float = 2.0

    @staticmethod
    def from_dict(name: str, payload: Dict) -> "MLPConfig":
        return MLPConfig(
            name=name,
            hidden_dims=list(payload.get("hidden_dims", [512, 256])),
            dropout=float(payload.get("dropout", 0.0)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            batch_size=int(payload.get("batch_size", 64)),
            max_epochs=int(payload.get("max_epochs", 100)),
            patience=int(payload.get("patience", 15)),
            loss_type=str(payload.get("loss_type", "ce")),
            use_class_weights=bool(payload.get("use_class_weights", False)),
            focal_gamma=float(payload.get("focal_gamma", 2.0)),
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1 - p_t) ** self.gamma
        return torch.mean(focal * ce)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    weights = len(labels) / np.maximum(num_classes * counts, 1.0)
    weights[counts == 0] = 1.0
    weights = weights / np.max(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _eval(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict:
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = F.softmax(logits, dim=1)
            losses.append(float(loss.item()))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64)
    preds_np = np.argmax(probs_np, axis=1)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "probs": probs_np,
        "labels": labels_np,
        "preds": preds_np,
        "macro_f1": macro_f1_from_predictions(labels_np, preds_np),
    }


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    config: MLPConfig,
    seed: int,
) -> Dict:
    _seed_everything(seed)

    train_x = train_x.astype(np.float32)
    dev_x = dev_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_y = train_y.astype(np.int64)
    dev_y = dev_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    num_classes = int(max(train_y.max(initial=0), dev_y.max(initial=0), test_y.max(initial=0))) + 1
    input_dim = train_x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=config.hidden_dims, dropout=config.dropout)
    model.to(device)

    class_weight = _compute_class_weights(train_y, num_classes) if config.use_class_weights else None

    if config.loss_type == "focal":
        criterion = FocalLoss(gamma=config.focal_gamma, class_weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device) if class_weight is not None else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    dev_ds = TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    best_state = None
    best_dev_f1 = float("-inf")
    best_epoch = -1
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(config.max_epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        dev_eval = _eval(model, dev_loader, device=device, criterion=criterion)
        epoch_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "dev_loss": dev_eval["loss"],
                "dev_macro_f1": dev_eval["macro_f1"],
            }
        )

        if dev_eval["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_eval["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(history)

    model.load_state_dict(best_state)

    test_eval = _eval(model, test_loader, device=device, criterion=criterion)

    return {
        "probs": test_eval["probs"],
        "labels": test_eval["labels"],
        "pred_labels": test_eval["preds"],
        "best_epoch": best_epoch,
        "best_dev_macro_f1": float(best_dev_f1),
        "history": history,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "class_weight": class_weight.numpy().tolist() if class_weight is not None else None,
        "state_dict": best_state,
    }
