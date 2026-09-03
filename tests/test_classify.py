import numpy as np
import pytest

from app import classify
from app.config import NODATA


def test_dequantize_unit_range():
    # 127 is the largest representable magnitude (-128 is NoData), so it maps
    # to (127/127.5)**2 ~= 0.992, not exactly 1.0.
    a = np.array([[127, -127, 0, 64]], dtype=np.int8)
    dq = classify.dequantize(a)
    assert dq.dtype == np.float32
    assert dq[0, 0] == pytest.approx(0.9922, abs=1e-3)
    assert dq[0, 1] == pytest.approx(-0.9922, abs=1e-3)
    assert dq[0, 2] == 0.0
    assert dq[0, 3] == pytest.approx((64 / 127.5) ** 2, abs=1e-4)  # sign preserved


def test_sample_points_shape_and_skips(window, points):
    class_to_idx = {"A": 0, "B": 1}
    X, y, used = classify.sample_points(window, points, class_to_idx)
    assert X.shape == (10, 64)
    assert y.shape == (10,)
    assert all(used)


def test_sample_points_skips_nodata_and_outside(window):
    from tests.conftest import make_point, SimplePoint

    pts = [
        make_point(window, 0, 0, "A"),        # NoData corner -> skipped
        make_point(window, 20, 5, "A"),       # valid
        SimplePoint(lat=0.0, lon=0.0, cls="B"),  # far outside -> skipped
    ]
    X, y, used = classify.sample_points(window, pts, {"A": 0, "B": 1})
    assert X.shape[0] == 1
    assert used == [False, True, False]


@pytest.mark.parametrize("clf", ["rf", "knn"])
def test_train_and_predict_separates_classes(window, points, clf):
    res = classify.train_and_predict(window, window, points, ["A", "B"], clf)
    assert res.labels.shape == (40, 40)
    # left half is class A (0), right half is class B (1)
    assert res.labels[20, 5] == 0
    assert res.labels[20, 30] == 1
    # NoData corner stays masked
    assert res.labels[0, 0] == -1
    # every predicted label is a valid class index or -1
    assert set(np.unique(res.labels)).issubset({-1, 0, 1})
    assert res.n_points_used == 10
    assert res.accuracy is None or 0.0 <= res.accuracy <= 1.0


def test_train_requires_two_classes(window):
    from tests.conftest import make_point

    only_a = [make_point(window, r, 5, "A") for r in (10, 15, 20)]
    with pytest.raises(ValueError):
        classify.train_and_predict(window, window, only_a, ["A", "B"], "rf")


def test_predict_can_apply_to_other_window(window, points):
    # A "different year": swap the two halves' signatures.
    other = window.array.copy()
    other[:, :20, :], other[:, 20:, :] = window.array[:, 20:, :], window.array[:, :20, :]
    from app.embeddings import EmbeddingWindow

    other_win = EmbeddingWindow(other, window.x0, window.y0, window.pix, window.epsg)
    res = classify.train_and_predict(window, other_win, points, ["A", "B"], "rf")
    # signatures flipped -> left now looks like B, right like A
    assert res.labels[20, 5] == 1
    assert res.labels[20, 30] == 0


def test_train_model_and_predict_window_match_wrapper(window, points):
    trained = classify.train_model(window, points, ["A", "B"], "rf")
    assert trained.n_classes == 2
    assert trained.n_points_used == 10
    assert trained.classifier == "rf"

    labels = classify.predict_window(trained, window)
    assert labels[20, 5] == 0 and labels[20, 30] == 1
    assert labels[0, 0] == -1  # nodata corner stays masked

    # The split must reproduce the one-shot wrapper exactly.
    res = classify.train_and_predict(window, window, points, ["A", "B"], "rf")
    assert np.array_equal(labels, res.labels)


def test_transition_map_ids_and_observed():
    k = 3
    a = np.array([[0, 1, 2], [0, -1, 2]], dtype=np.int16)
    b = np.array([[0, 2, 0], [1, 0, 2]], dtype=np.int16)
    ids, observed, n_comparable = classify.transition_map(a, b, k)

    assert ids[0, 0] == -1          # 0 -> 0 unchanged
    assert ids[1, 2] == -1          # 2 -> 2 unchanged
    assert ids[1, 1] == -1          # nodata in A
    assert ids[0, 1] == 1 * k + 2   # 1 -> 2
    assert ids[0, 2] == 2 * k + 0   # 2 -> 0
    assert ids[1, 0] == 0 * k + 1   # 0 -> 1

    assert {(f, t) for f, t, _ in observed} == {(1, 2), (2, 0), (0, 1)}
    assert all(px == 1 for _, _, px in observed)
    assert n_comparable == 5        # 6 pixels, 1 nodata in A (changed + unchanged)
