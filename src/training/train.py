"""Generic training loop for all RNA -> protein model variants.

One fit_model() replaces the five near-identical training-loop copies in
nb05-nb09. Works with any (model, penalty_fn) pair from architectures.build_variant.

Device pattern (matches nb06 onward): model moves to `train_device` only for
the training pass each epoch; validation runs on CPU. The model returned is
left on CPU, ready for CPU-only evaluation.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, Y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a DataLoader over the rows of X, Y selected by idx.

    drop_last=shuffle so training batches never hit batch_size=1 (breaks
    BatchNorm, though these architectures don't currently use it -- kept for
    consistency with the notebooks this replaces).
    """
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    penalty_fn: Callable[[nn.Module], torch.Tensor],
    lr: 1e-3,
    num_epochs: int,
    patience: int,
    train_device: torch.device,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
    log_every: int = 10,
) -> tuple[nn.Module, dict]:
    """Train with MSE + penalty_fn(model) on train_device; validate on CPU each epoch.

    Early stopping on validation loss. The best-validation state dict is
    reloaded before returning, and the model is left on CPU.

    penalty_fn takes only the model (e.g. the closure returned by
    architectures.build_variant) and returns a scalar penalty tensor.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.to(train_device)
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(train_device), yb.to(train_device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb) + penalty_fn(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.to("cpu")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.to("cpu")
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return model, history
