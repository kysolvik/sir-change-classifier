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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from . import config
from .embeddings import EmbeddingWindow
from .geo import point_rowcol

NODATA = config.NODATA
_PREDICT_CHUNK = 250_000
_CV_MAX_FOLDS = 5
_CV_REPEATS = 3


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
    per_class_accuracy: dict[int, float] = field(default_factory=dict)


@dataclass
class Evaluation:
    """Cross-validated skill on the training points."""

    accuracy: float | None  # None when too few points per class to cross-validate
    per_class: dict[int, float]  # class idx -> share of its points predicted right
    n_points: int


@dataclass
class TrainedModel:
    """A fitted classifier plus the metadata needed to apply and report on it."""

    model: object
    classifier: str
    accuracy: float | None
    n_classes: int
    n_points_used: int
    n_points_skipped: int
    used_flags: list[bool] = field(default_factory=list)  # per input point
    per_class_accuracy: dict[int, float] = field(default_factory=dict)


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
        if _masked(vec, win):
            used.append(False)
            continue
        xs.append(vec)
        ys.append(class_to_idx[p.cls])
        used.append(True)
    n_bands = win.array.shape[-1]
    dtype = win.array.dtype
    X = np.asarray(xs, dtype=dtype) if xs else np.empty((0, n_bands), dtype)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, used


def _is_int8(win: EmbeddingWindow) -> bool:
    """AEF windows are int8 with a NODATA sentinel; raw imagery is float w/ NaN."""
    return win.array.dtype == np.int8


def _masked(flat: np.ndarray, win: EmbeddingWindow) -> np.ndarray:
    """Per-row mask: int8 uses the NODATA sentinel, float uses NaN."""
    if _is_int8(win):
        return np.any(flat == NODATA, axis=-1)
    return np.any(np.isnan(flat), axis=-1)


def _build_model(classifier: str, n_samples: int, n_jobs: int = -1):
    if classifier == "knn":
        k = min(config.KNN_NEIGHBORS, max(1, n_samples))
        return KNeighborsClassifier(n_neighbors=k)
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        n_jobs=n_jobs,
        random_state=0,
    )


def cross_validate(classifier: str, Xf: np.ndarray, y: np.ndarray) -> Evaluation:
    """Repeated stratified k-fold accuracy, overall and per class.

    Every point is held out once per repeat and predicted by a model that never
    saw it. Folds depend only on ``y`` (fixed seed), so two feature sources
    evaluated on the same points are scored on identical splits.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return Evaluation(None, {}, int(y.size))
    n_splits = int(min(_CV_MAX_FOLDS, counts.min()))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=_CV_REPEATS, random_state=0)
    right = np.zeros(y.size)
    tries = np.zeros(y.size)
    for tr, te in cv.split(Xf, y):
        # Tiny fits: one thread beats joblib's start-up cost.
        model = _build_model(classifier, tr.size, n_jobs=1)
        model.fit(Xf[tr], y[tr])
        right[te] += model.predict(Xf[te]) == y[te]
        tries[te] += 1
    per_class = {int(c): float(right[y == c].sum() / tries[y == c].sum()) for c in classes}
    return Evaluation(float(right.sum() / tries.sum()), per_class, int(y.size))


def _predict_full(model, win: EmbeddingWindow, classifier: str) -> np.ndarray:
    h, w, n_bands = win.array.shape
    flat = win.array.reshape(-1, n_bands)
    masked = _masked(flat, win)
    out = np.full(flat.shape[0], -1, dtype=np.int16)

    valid_idx = np.nonzero(~masked)[0]
    for start in range(0, valid_idx.size, _PREDICT_CHUNK):
        idx = valid_idx[start : start + _PREDICT_CHUNK]
        block = flat[idx]
        if classifier == "knn" and _is_int8(win):
            block = dequantize(block)
        out[idx] = model.predict(block).astype(np.int16)
    return out.reshape(h, w)


def train_model(
    train_win: EmbeddingWindow,
    points,
    classes: list[str],
    classifier: str = "rf",
) -> TrainedModel:
    """Sample the training-year embedding at each point and fit a classifier."""
    class_to_idx = {name: i for i, name in enumerate(classes)}
    X, y, used = sample_points(train_win, points, class_to_idx)

    if X.shape[0] < config.MIN_POINTS_TOTAL or np.unique(y).size < 2:
        raise ValueError(
            "Need at least two usable points in at least two classes "
            "(check that points sit inside the box on valid pixels)."
        )

    Xf = _model_space(X, train_win, classifier)
    model = _build_model(classifier, X.shape[0])
    evaluation = cross_validate(classifier, Xf, y)
    model.fit(Xf, y)

    return TrainedModel(
        model=model,
        classifier=classifier,
        accuracy=evaluation.accuracy,
        per_class_accuracy=evaluation.per_class,
        n_classes=len(classes),
        n_points_used=int(X.shape[0]),
        n_points_skipped=int(len(used) - X.shape[0]),
        used_flags=used,
    )


def _model_space(X: np.ndarray, win: EmbeddingWindow, classifier: str) -> np.ndarray:
    """KNN needs AEF int8 dequantized to its true unit-vector space."""
    return dequantize(X) if (classifier == "knn" and _is_int8(win)) else X


@dataclass
class SourceEvaluation:
    evaluation: Evaluation
    n_points_usable: int  # usable in this source alone (before intersecting)


def compare_sources(
    wins: dict[str, EmbeddingWindow],
    points,
    classes: list[str],
    classifier: str = "rf",
) -> tuple[dict[str, SourceEvaluation], int]:
    """Cross-validate the same classifier on each feature source, fairly.

    Only points usable in *every* source are scored (a point on a cloud gap in
    one source is dropped from all), and the folds are identical, so accuracy
    differences come from the features alone. Returns per-source results and
    the number of shared points.
    """
    class_to_idx = {name: i for i, name in enumerate(classes)}
    sampled = {sid: sample_points(win, points, class_to_idx) for sid, win in wins.items()}
    used = np.ones(len(points), dtype=bool)
    for _, _, flags in sampled.values():
        used &= np.asarray(flags, dtype=bool)
    shared = np.nonzero(used)[0]
    y_all = np.asarray([class_to_idx[p.cls] for p in points], dtype=np.int64)
    y = y_all[shared]
    if shared.size < config.MIN_POINTS_TOTAL or np.unique(y).size < 2:
        raise ValueError(
            "Need at least two points in at least two classes that are usable in "
            "every feature source (points on cloud gaps or outside the box are skipped)."
        )

    out = {}
    for sid, (X, _, flags) in sampled.items():
        rows = np.searchsorted(np.nonzero(flags)[0], shared)  # X row of each shared point
        Xf = _model_space(X[rows], wins[sid], classifier)
        out[sid] = SourceEvaluation(cross_validate(classifier, Xf, y), int(X.shape[0]))
    return out, int(shared.size)


def predict_window(trained: TrainedModel, win: EmbeddingWindow) -> np.ndarray:
    """Apply a trained model to a full window, returning an (H, W) int16 label map."""
    return _predict_full(trained.model, win, trained.classifier)


def train_and_predict(
    train_win: EmbeddingWindow,
    target_win: EmbeddingWindow,
    points,
    classes: list[str],
    classifier: str = "rf",
) -> ClassifyResult:
    trained = train_model(train_win, points, classes, classifier)
    labels = predict_window(trained, target_win)

    idx_vals, idx_counts = np.unique(labels[labels >= 0], return_counts=True)
    counts = {int(i): int(c) for i, c in zip(idx_vals, idx_counts)}
    return ClassifyResult(
        labels=labels,
        n_classes=trained.n_classes,
        accuracy=trained.accuracy,
        class_pixel_counts=counts,
        n_points_used=trained.n_points_used,
        n_points_skipped=trained.n_points_skipped,
        used_flags=trained.used_flags,
        per_class_accuracy=trained.per_class_accuracy,
    )


def transition_map(
    labels_a: np.ndarray, labels_b: np.ndarray, k: int
) -> tuple[np.ndarray, list[tuple[int, int, int]], int]:
    """Encode per-pixel class change between two label maps.

    A pixel that is valid (>= 0) in both years and whose class changed gets id
    ``from * k + to``; unchanged or nodata pixels get ``-1`` (transparent).
    Returns ``(ids, observed, n_comparable)`` where ``observed`` is the list of
    ``(from, to, pixels)`` sorted by area desc (the legend) and ``n_comparable``
    is the count of pixels valid in *both* years — the denominator for "what
    fraction of the area changed" (changed + unchanged, excluding nodata).
    """
    comparable = (labels_a >= 0) & (labels_b >= 0)
    changed = comparable & (labels_a != labels_b)
    ids = np.full(labels_a.shape, -1, dtype=np.int32)
    a = labels_a.astype(np.int32)
    b = labels_b.astype(np.int32)
    ids[changed] = a[changed] * k + b[changed]

    uniq, counts = np.unique(ids[changed], return_counts=True)
    order = np.argsort(-counts)
    observed = [
        (int(u // k), int(u % k), int(c))
        for u, c in zip(uniq[order], counts[order])
    ]
    return ids, observed, int(comparable.sum())
