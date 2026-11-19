"""API tests. The embedding source is stubbed (no network): the cache's reader
returns the synthetic separable window, so the real classify/colorize pipeline
and endpoint validation are exercised end to end."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.embeddings import EmbeddingWindow
from tests.conftest import make_point, make_window
from tests.test_cache import InMemoryBackend

WINDOW = make_window()


def _float_window():
    """A raw-Landsat-style float composite: 6 bands, left=A/right=B, NaN corner.
    Same geotransform as WINDOW so _payload's points land on the right halves."""
    arr = np.full((40, 40, 6), 0.1, dtype=np.float32)
    arr[:, :20, :3] = 0.8
    arr[:, 20:, 3:] = 0.8
    arr[0:2, 0:2, :] = np.nan
    return EmbeddingWindow(arr, WINDOW.x0, WINDOW.y0, WINDOW.pix, WINDOW.epsg)


FLOAT_WINDOW = _float_window()


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
    main.cache.clear_hot()
    main.limiter.reset()
    main.classify_cap._stamps.clear()
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
    main.cache.clear_hot()
    main.limiter.reset()
    main.classify_cap._stamps.clear()
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


# --- abuse / cost guards ---------------------------------------------------
def _req(xff=None, host="10.0.0.1"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return main.Request({"type": "http", "headers": headers, "client": (host, 1234)})


def test_client_ip_trusts_rightmost_forwarded_entry():
    # The left entries are client-supplied (spoofable); Cloud Run appends the real one.
    assert main.client_ip(_req("6.6.6.6, 1.2.3.4")) == "1.2.3.4"
    assert main.client_ip(_req("1.2.3.4")) == "1.2.3.4"


def test_client_ip_falls_back_without_header():
    assert main.client_ip(_req()) == "10.0.0.1"


def test_global_cap_returns_503(client, monkeypatch):
    monkeypatch.setattr(main.classify_cap, "per_hour", 2)
    assert client.post("/api/classify", json=_payload()).status_code == 200
    assert client.post("/api/classify", json=_payload()).status_code == 200
    r = client.post("/api/classify", json=_payload())
    assert r.status_code == 503
    # Rotating IPs doesn't help: the cap is global.
    r = client.post("/api/classify", json=_payload(), headers={"X-Forwarded-For": "9.9.9.9"})
    assert r.status_code == 503


def test_default_limit_covers_static_files_per_ip(client, monkeypatch):
    monkeypatch.setattr(main, "_default_limit", main.parse_limit("3/minute"))
    for _ in range(3):
        assert client.get("/", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.get("/", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    # A different client is unaffected.
    assert client.get("/", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_static_files_gzipped_and_cacheable(client):
    r = client.get("/app.js", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert "max-age" in r.headers["cache-control"]


@pytest.fixture
def landsat_client(monkeypatch):
    async def fake_aef(box, year):
        return WINDOW

    async def fake_landsat(box, year):
        return FLOAT_WINDOW

    async def fake_get_index():
        return None

    monkeypatch.setattr(main.embeddings, "get_index", fake_get_index)
    monkeypatch.setattr(main.cache, "_reader", fake_aef)
    monkeypatch.setattr(main.cache, "_landsat_reader", fake_landsat)
    monkeypatch.setattr(main.cache, "backend", InMemoryBackend())
    main.cache.clear_hot()
    with TestClient(main.app) as c:
        yield c


def test_classify_landsat_happy_path(landsat_client):
    # _payload's 37.78,-122.44 is the SF preset, so raw Landsat is allowed.
    r = landsat_client.post("/api/classify", json=_payload(feature_source="landsat"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["class_pixel_counts"]) == {"A", "B"}
    assert body["n_points_used"] == 6


def test_classify_landsat_rejected_off_preset(landsat_client):
    r = landsat_client.post(
        "/api/classify", json=_payload(feature_source="landsat", lat=0.0, lon=0.0)
    )
    assert r.status_code == 400
    assert "example places" in r.json()["detail"]


# --- multi-year Landsat, feature comparison, imagery ------------------------
def test_config_lists_feature_sources(client):
    body = client.get("/api/config").json()
    by_id = {f["id"]: f for f in body["feature_sources"]}
    assert by_id["aef"]["min_year"] == 2017
    assert by_id["landsat"]["min_year"] == 1984
    assert by_id["landsat"]["imagery"] and not by_id["aef"]["imagery"]
    assert {s["id"] for s in body["imagery_styles"]} >= {"natural", "false"}


def test_landsat_accepts_pre_aef_years(landsat_client):
    r = landsat_client.post(
        "/api/compare",
        json=_compare_payload(feature_source="landsat", training_year=1990,
                              year_a=1985, year_b=2025),
    )
    assert r.status_code == 200, r.text
    assert r.json()["year_a"] == 1985


def test_aef_rejects_pre_2017_years(landsat_client):
    r = landsat_client.post("/api/classify", json=_payload(training_year=1990))
    assert r.status_code == 422


def _eval_payload(**overrides):
    payload = _payload()
    payload.pop("target_year")
    payload.update(overrides)
    return payload


def test_evaluate_scores_every_source_on_shared_points(landsat_client):
    r = landsat_client.post("/api/evaluate", json=_eval_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["source"] for x in body["results"]] == ["aef", "landsat"]
    assert body["n_points_shared"] == 6 and body["skipped"] == []
    for res in body["results"]:
        assert res["accuracy"] == 1.0  # the synthetic halves are separable
        assert set(res["per_class"]) == {"A", "B"}


def test_evaluate_skips_sources_not_covering_year(landsat_client):
    r = landsat_client.post("/api/evaluate", json=_eval_payload(training_year=1995))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["source"] for x in body["results"]] == ["landsat"]
    assert body["skipped"][0]["source"] == "aef"


def test_evaluate_skips_preset_only_source_off_preset(landsat_client):
    r = landsat_client.post("/api/evaluate", json=_eval_payload(lat=0.0, lon=0.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["source"] for x in body["results"]] == ["aef"]
    assert body["skipped"][0]["source"] == "landsat"
    assert "example places" in body["skipped"][0]["reason"]
    # Pre-2017 off-preset: no source is left at all.
    r = landsat_client.post(
        "/api/evaluate", json=_eval_payload(lat=0.0, lon=0.0, training_year=1995)
    )
    assert r.status_code == 400
    assert "No feature source" in r.json()["detail"]


def test_imagery_renders_landsat(landsat_client):
    r = landsat_client.get("/api/imagery", params={"lat": 37.78, "lon": -122.44, "year": 1990})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert body["year"] == 1990 and body["style"] == "natural"
    assert 0.99 < body["coverage"] < 1.0  # the NaN corner


@pytest.mark.parametrize(
    "params, code",
    [
        ({"year": 1990, "style": "nope"}, 400),
        ({"year": 1970}, 422),
        ({"year": 2020, "source": "aef"}, 400),  # embeddings aren't a picture
        ({"year": 2020, "lat": 0.0, "lon": 0.0}, 400),  # off-preset
    ],
)
def test_imagery_rejects_bad_requests(landsat_client, params, code):
    q = {"lat": 37.78, "lon": -122.44, **params}
    assert landsat_client.get("/api/imagery", params=q).status_code == code
