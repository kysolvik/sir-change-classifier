import math

from app import config, geo


def test_utm_epsg_hemispheres():
    assert geo.utm_epsg(-122.44, 37.78) == 32610   # San Francisco, zone 10N
    assert geo.utm_epsg(31.1, 30.7) == 32636        # Nile Delta, zone 36N
    assert geo.utm_epsg(-62.7, -10.2) == 32720      # Rondonia, zone 20S


def test_box_is_requested_size():
    box = geo.build_box(-122.44, 37.78, size_km=10, snap_m=0)
    assert box.width_px == 1000
    assert box.height_px == 1000
    assert math.isclose(box.xmax - box.xmin, 10_000, abs_tol=1)


def test_box_snapping_collapses_nearby_points():
    a = geo.build_box(-122.4400, 37.7800, snap_m=1000)
    b = geo.build_box(-122.4405, 37.7803, snap_m=1000)  # ~50 m away
    assert a.key() == b.key()


def test_lonlat_bounds_bracket_center():
    box = geo.build_box(-122.44, 37.78)
    min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()
    assert min_lon < box.center_lon < max_lon
    assert min_lat < box.center_lat < max_lat


def test_point_rowcol_roundtrip_and_bounds():
    from app.embeddings import EmbeddingWindow
    import numpy as np

    win = EmbeddingWindow(np.zeros((100, 100, 64), np.int8), 500005.0, 4185005.0, 10.0, 32610)
    # centre of pixel (row=30, col=40)
    lon, lat = geo.utm_to_lonlat(win.epsg, win.x0 + 40 * 10, win.y0 - 30 * 10)
    assert geo.point_rowcol(win, lon, lat) == (30, 40)
    # far away -> None
    assert geo.point_rowcol(win, 0.0, 0.0) is None
