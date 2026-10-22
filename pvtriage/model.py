"""Small CNN that classifies a thermal module crop and emits an embedding.

The embedding is what gets indexed in DuckDB, so the same forward pass serves
both the classifier and the precedent search.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """Two input channels: absolute level and level relative to the module median.

    The second channel makes the network look at contrast inside the module,
    which is what a thermographer reads, instead of the absolute gray level that
    depends on camera gain and ambient conditions.
    """
    x = torch.from_numpy(images.astype(np.float32) / 255.0)
    median = x.flatten(1).median(dim=1).values.view(-1, 1, 1)
    relative = (x - median) * 4.0
    return torch.stack([x - 0.5, relative], dim=1)


class ThermalNet(nn.Module):
    def __init__(self, n_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(2, 32),
            nn.MaxPool2d(2),  # 20x12
            block(32, 64),
            nn.MaxPool2d(2),  # 10x6
            block(64, 128),
        )
        self.embed = nn.Sequential(nn.Dropout(0.2), nn.Linear(128 * 2, embedding_dim))
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        pooled = torch.cat([F.adaptive_avg_pool2d(h, 1), F.adaptive_max_pool2d(h, 1)], dim=1).flatten(1)
        z = self.embed(pooled)
        return self.head(F.relu(z)), F.normalize(z, dim=1)


@dataclass
class TrainResult:
    model: ThermalNet
    history: list[dict[str, float]]
    seconds: float


def _augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-2)
    return x


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse square root frequency: lifts rare classes without letting them dominate."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(
    images: np.ndarray,
    labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    embedding_dim: int,
    seed: int,
    on_epoch=None,
) -> TrainResult:
    torch.manual_seed(seed)
    model = ThermalNet(n_classes, embedding_dim)
    x_train, y_train = to_tensor(images), torch.from_numpy(labels)
    x_val, y_val = to_tensor(val_images), torch.from_numpy(val_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * int(np.ceil(len(x_train) / batch_size)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels, n_classes), label_smoothing=0.05)
    generator = torch.Generator().manual_seed(seed)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            logits, _ = model(_augment(x_train[idx]))
            loss = loss_fn(logits, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        val_probs, _ = predict(model, x_val)
        val_acc = float((val_probs.argmax(1) == y_val.numpy()).mean()) if len(y_val) else 0.0
        row = {"epoch": epoch + 1, "train_loss": total / len(x_train), "val_accuracy": val_acc}
        history.append(row)
        log.info("epoch %d loss %.4f val_acc %.4f", epoch + 1, row["train_loss"], val_acc)
        if on_epoch:
            on_epoch(row)
    return TrainResult(model, history, time.perf_counter() - start)


@torch.no_grad()
def predict(model: ThermalNet, x: torch.Tensor | np.ndarray, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, unit length embeddings) as numpy arrays."""
    if isinstance(x, np.ndarray):
        x = to_tensor(x)
    model.eval()
    probs, embs = [], []
    for i in range(0, len(x), batch_size):
        logits, z = model(x[i : i + batch_size])
        probs.append(F.softmax(logits, dim=1).numpy())
        embs.append(z.numpy())
    if not probs:
        return np.empty((0, model.head.out_features)), np.empty((0, model.head.in_features))
    return np.concatenate(probs), np.concatenate(embs)
