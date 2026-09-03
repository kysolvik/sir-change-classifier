"""API tests. The embedding source is stubbed (no network): the cache's reader
returns the synthetic separable window, so the real classify/colorize pipeline
and endpoint validation are exercised end to end."""
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.embeddings import EmbeddingWindow
from tests.conftest import make_point, make_window
from tests.test_cache import InMemoryBackend

WINDOW = make_window()


def _swapped(win):
    """A 'different year': the two halves' signatures swapped (A<->B)."""
    arr = win.array.copy()
    arr[:, :20, :], arr[:, 20:, :] = win.array[:, 20:, :], win.array[:, :20, :]
    return EmbeddingWindow(arr, win.x0, win.y0, win.pix, win.epsg)


SWAPPED = _swapped(WINDOW)


@pytest.fixture
def client(monkeypatch):
    async def fake_reader(box, year):
        return WINDOW

    async def fake_get_index():
        return None

    monkeypatch.setattr(main.embeddings, "get_index", fake_get_index)
    monkeypatch.setattr(main.cache, "_reader", fake_reader)
    monkeypatch.setattr(main.cache, "backend", InMemoryBackend())
    main.cache._inproc.clear()
    with TestClient(main.app) as c:
        yield c


def _payload(**overrides):
    a = [make_point(WINDOW, r, c, "A") for r, c in [(10, 5), (20, 3), (30, 6)]]
    b = [make_point(WINDOW, r, c, "B") for r, c in [(10, 30), (20, 28), (30, 31)]]
    payload = {
        "lat": 37.78,
        "lon": -122.44,
        "training_year": 2025,
        "target_year": 2025,
        "classifier": "rf",
        "classes": [{"name": "A", "color": "#ff0000"}, {"name": "B", "color": "#0000ff"}],
        "points": [{"class": p.cls, "lat": p.lat, "lon": p.lon} for p in a + b],
    }
    payload.update(overrides)
    return payload


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["box_size_km"] > 0
    assert body["default_year"] in body["years"]
    assert len(body["presets"]) >= 1


def test_classify_happy_path(client):
    r = client.post("/api/classify", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert len(body["bounds"]) == 2
    assert set(body["class_pixel_counts"]) == {"A", "B"}
    assert body["n_points_used"] == 6


def test_classify_rejects_bad_year(client):
    r = client.post("/api/classify", json=_payload(training_year=2030))
    assert r.status_code == 422


def test_classify_rejects_unknown_class_point(client):
    bad = _payload()
    bad["points"].append({"class": "Z", "lat": 37.78, "lon": -122.44})
    r = client.post("/api/classify", json=bad)
    assert r.status_code == 400


def test_classify_rejects_empty_classes(client):
    r = client.post("/api/classify", json=_payload(classes=[]))
    assert r.status_code == 422


@pytest.fixture
def compare_client(monkeypatch):
    # year_b (2018) returns the signature-swapped window, so every valid pixel
    # flips class -> known A<->B transitions.
    async def fake_reader(box, year):
        return SWAPPED if year == 2018 else WINDOW

    async def fake_get_index():
        return None

    monkeypatch.setattr(main.embeddings, "get_index", fake_get_index)
    monkeypatch.setattr(main.cache, "_reader", fake_reader)
    monkeypatch.setattr(main.cache, "backend", InMemoryBackend())
    main.cache._inproc.clear()
    with TestClient(main.app) as c:
        yield c


def _compare_payload(**overrides):
    payload = _payload()
    del payload["target_year"]
    payload.update({"year_a": 2025, "year_b": 2018})
    payload.update(overrides)
    return payload


def test_compare_happy_path(compare_client):
    r = compare_client.post("/api/compare", json=_compare_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert body["year_a"] == 2025 and body["year_b"] == 2018
    pairs = {(t["from"], t["to"]) for t in body["transitions"]}
    assert pairs == {("A", "B"), ("B", "A")}
    assert all(t["pixels"] > 0 and t["color"].startswith("#") for t in body["transitions"])
    # Denominator for "% of area changed": comparable >= the changed pixels.
    changed = sum(t["pixels"] for t in body["transitions"])
    assert body["compared_pixels"] >= changed > 0


def test_compare_rejects_bad_year(compare_client):
    r = compare_client.post("/api/compare", json=_compare_payload(year_b=2030))
    assert r.status_code == 422


def test_compare_rejects_unknown_class_point(compare_client):
    bad = _compare_payload()
    bad["points"].append({"class": "Z", "lat": 37.78, "lon": -122.44})
    r = compare_client.post("/api/compare", json=bad)
    assert r.status_code == 400
