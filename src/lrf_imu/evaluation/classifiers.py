"""Random-Forest and lazy Torch CNN classifiers used by the active evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
import random
from typing import Any, Optional

import numpy as np

from .scenarios import ensure_nct, flatten_windows


@dataclass(frozen=True)
class RandomForestSpec:
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class CNNTrainingSpec:
    conv_filters: tuple[int, int, int] = (32, 64, 128)
    fc_hidden: tuple[int, int] = (256, 128)
    kernel_size: int = 5
    padding: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 80
    patience: int = 10
    batch_size: int = 64
    validation_fraction: float = 0.20
    class_weights: bool = False
    seed: int = 42


def seed_cnn_run(seed: int) -> None:
    """Seed once at the ordered scenario-run boundary, as the active source does."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CNN evaluation requires Torch; install the training extra") from exc
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict_random_forest(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    *,
    spec: RandomForestSpec = RandomForestSpec(),
) -> np.ndarray:
    """Fit/predict using only the three source-explicit RF arguments."""

    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Random-Forest evaluation requires scikit-learn; install the evaluation extra"
        ) from exc
    classifier = RandomForestClassifier(
        n_estimators=spec.n_estimators,
        random_state=spec.random_state,
        n_jobs=spec.n_jobs,
    )
    classifier.fit(flatten_windows(train_windows), np.asarray(train_labels, dtype=np.int64))
    return np.asarray(classifier.predict(flatten_windows(test_windows)), dtype=np.int64)


def stratified_train_validation_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Mirror source StratifiedShuffleSplit, including its permutation fallback."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = y.size
    if n < 10 or validation_fraction <= 0.0:
        return np.arange(n, dtype=np.int64), None
    try:
        from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import-untyped]

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        train, validation = next(splitter.split(np.zeros((n, 1)), y))
        return train.astype(np.int64), validation.astype(np.int64)
    except Exception:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        n_validation = max(1, int(round(validation_fraction * n)))
        return indices[n_validation:], indices[:n_validation]


def cnn_state_geometry(
    in_channels: int,
    sequence_length: int,
    num_classes: int = 4,
    *,
    spec: CNNTrainingSpec = CNNTrainingSpec(),
) -> dict[str, tuple[int, ...]]:
    """Expected learnable tensor geometry of the active 1D CNN."""

    f1, f2, f3 = spec.conv_filters
    h1, h2 = spec.fc_hidden
    pooled = sequence_length // 2
    return {
        "conv1.weight": (f1, in_channels, 5), "conv1.bias": (f1,),
        "conv2.weight": (f2, f1, 5), "conv2.bias": (f2,),
        "conv3.weight": (f3, f2, 5), "conv3.bias": (f3,),
        "fc1.weight": (h1, f3 * pooled), "fc1.bias": (h1,),
        "fc2.weight": (h2, h1), "fc2.bias": (h2,),
        "fc3.weight": (num_classes, h2), "fc3.bias": (num_classes,),
    }


def build_har_cnn(
    in_channels: int,
    sequence_length: int,
    num_classes: int = 4,
    *,
    spec: CNNTrainingSpec = CNNTrainingSpec(),
):
    """Build the historical CNN without importing Torch at package import time."""

    try:
        import torch  # noqa: F401
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CNN evaluation requires Torch; install the training extra") from exc
    f1, f2, f3 = spec.conv_filters
    h1, h2 = spec.fc_hidden

    class HARClassifier1DCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, f1, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv1d(f1, f2, kernel_size=5, stride=1, padding=2)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv1d(f2, f3, kernel_size=5, stride=1, padding=2)
            self.fc1 = nn.Linear(f3 * (sequence_length // 2), h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, num_classes)
            self.dropout = nn.Dropout(p=spec.dropout)
            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = inputs
            if x.ndim != 3:
                raise ValueError("CNN inputs must be N,C,T or N,T,C")
            if x.shape[1] not in (3, 6) and x.shape[2] in (3, 6):
                x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            return self.fc3(x)

    return HARClassifier1DCNN()


def predict_cnn(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    *,
    channels: int,
    sequence_length: int,
    num_classes: int = 4,
    device: str = "cpu",
    spec: CNNTrainingSpec = CNNTrainingSpec(),
) -> np.ndarray:
    """Train with the source seed/split/shuffle/early-stopping protocol."""

    try:
        import torch
        from sklearn.metrics import f1_score  # type: ignore[import-untyped]
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CNN evaluation requires Torch and scikit-learn") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_x = ensure_nct(np.asarray(train_windows, dtype=np.float32), channels)
    test_x = ensure_nct(np.asarray(test_windows, dtype=np.float32), channels)
    train_y = np.asarray(train_labels, dtype=np.int64)
    train_idx, validation_idx = stratified_train_validation_indices(
        train_y, validation_fraction=spec.validation_fraction, seed=spec.seed
    )
    if validation_idx is None:
        validation_idx = train_idx
    train_dataset = TensorDataset(
        torch.from_numpy(train_x[train_idx]), torch.from_numpy(train_y[train_idx])
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(train_x[validation_idx]), torch.from_numpy(train_y[validation_idx])
    )
    generator = torch.Generator().manual_seed(spec.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=spec.batch_size, shuffle=True,
        num_workers=0, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=spec.batch_size, shuffle=False, num_workers=0
    )
    model = build_har_cnn(
        channels, sequence_length, num_classes, spec=spec
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    best_state: Optional[dict[str, Any]] = None
    best_validation = -1.0
    bad_epochs = 0
    for _ in range(spec.max_epochs):
        model.train()
        for windows, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(windows.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        truth: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        with torch.no_grad():
            for windows, labels in validation_loader:
                predicted.append(model(windows.to(device)).argmax(dim=1).cpu().numpy())
                truth.append(labels.numpy())
        validation_f1 = f1_score(
            np.concatenate(truth), np.concatenate(predicted), average="macro",
            labels=list(range(num_classes)), zero_division=0
        )
        if validation_f1 > best_validation + 1e-6:
            best_validation = float(validation_f1)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= spec.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_x).to(device)).argmax(dim=1).cpu().numpy()


__all__ = [
    "CNNTrainingSpec", "RandomForestSpec", "build_har_cnn",
    "cnn_state_geometry", "predict_cnn", "predict_random_forest",
    "seed_cnn_run", "stratified_train_validation_indices",
]
