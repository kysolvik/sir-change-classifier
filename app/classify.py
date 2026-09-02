"""Train a simple classifier on sampled points and apply it to a full window.

The AlphaEarth embedding of a pixel is a unit vector once dequantized, so even a
few points per class separate well. Random forest (the default) is invariant to
the monotonic int8->float dequantization, so it runs on the raw int8; KNN uses
Euclidean distance, so for it we dequantize to the true unit-vector space first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from . import config
from .embeddings import EmbeddingWindow
from .geo import point_rowcol

NODATA = config.NODATA
_PREDICT_CHUNK = 250_000


def dequantize(a: np.ndarray) -> np.ndarray:
    """Map raw int8 embedding values to the [-1, 1] unit-vector space."""
    f = a.astype(np.float32)
    return np.sign(f) * (f / 127.5) ** 2


@dataclass
class ClassifyResult:
    labels: np.ndarray  # (H, W) int16; -1 = nodata/masked
    n_classes: int
    accuracy: float | None
    class_pixel_counts: dict[int, int]
    n_points_used: int
    n_points_skipped: int
    used_flags: list[bool] = field(default_factory=list)  # per input point


def sample_points(win: EmbeddingWindow, points, class_to_idx):
    """Return (X, y, used_flags) sampling the embedding at each point.

    Points outside the box or on masked (NoData) pixels are skipped.
    ``points`` is a sequence of objects with ``.lat``, ``.lon`` and ``.cls``.
    """
    xs, ys, used = [], [], []
    for p in points:
        rc = point_rowcol(win, p.lon, p.lat)
        if rc is None:
            used.append(False)
            continue
        vec = win.array[rc[0], rc[1]]
        if np.any(vec == NODATA):
            used.append(False)
            continue
        xs.append(vec)
        ys.append(class_to_idx[p.cls])
        used.append(True)
    X = np.asarray(xs, dtype=np.int8) if xs else np.empty((0, 64), np.int8)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, used


def _build_model(classifier: str, n_samples: int):
    if classifier == "knn":
        k = min(config.KNN_NEIGHBORS, max(1, n_samples))
        return KNeighborsClassifier(n_neighbors=k)
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        n_jobs=-1,
        random_state=0,
    )


def _accuracy(model, Xf, y) -> float | None:
    """Cheap cross-validated accuracy for student feedback (best effort)."""
    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    if len(counts) < 2 or min_class < 2:
        return None
    cv = min(3, min_class)
    try:
        return float(cross_val_score(model, Xf, y, cv=cv).mean())
    except Exception:
        return None


def _predict_full(model, win: EmbeddingWindow, classifier: str) -> np.ndarray:
    h, w, _ = win.array.shape
    flat = win.array.reshape(-1, 64)
    masked = np.any(flat == NODATA, axis=1)
    out = np.full(flat.shape[0], -1, dtype=np.int16)

    valid_idx = np.nonzero(~masked)[0]
    for start in range(0, valid_idx.size, _PREDICT_CHUNK):
        idx = valid_idx[start : start + _PREDICT_CHUNK]
        block = flat[idx]
        if classifier == "knn":
            block = dequantize(block)
        out[idx] = model.predict(block).astype(np.int16)
    return out.reshape(h, w)


def train_and_predict(
    train_win: EmbeddingWindow,
    target_win: EmbeddingWindow,
    points,
    classes: list[str],
    classifier: str = "rf",
) -> ClassifyResult:
    class_to_idx = {name: i for i, name in enumerate(classes)}
    X, y, used = sample_points(train_win, points, class_to_idx)

    if X.shape[0] < config.MIN_POINTS_TOTAL or np.unique(y).size < 2:
        raise ValueError(
            "Need at least two usable points in at least two classes "
            "(check that points sit inside the box on valid pixels)."
        )

    Xf = dequantize(X) if classifier == "knn" else X
    model = _build_model(classifier, X.shape[0])
    accuracy = _accuracy(_build_model(classifier, X.shape[0]), Xf, y)
    model.fit(Xf, y)

    labels = _predict_full(model, target_win, classifier)

    idx_vals, idx_counts = np.unique(labels[labels >= 0], return_counts=True)
    counts = {int(i): int(c) for i, c in zip(idx_vals, idx_counts)}
    return ClassifyResult(
        labels=labels,
        n_classes=len(classes),
        accuracy=accuracy,
        class_pixel_counts=counts,
        n_points_used=int(X.shape[0]),
        n_points_skipped=int(len(used) - X.shape[0]),
        used_flags=used,
    )
