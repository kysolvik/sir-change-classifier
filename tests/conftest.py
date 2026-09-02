"""Shared test fixtures: a synthetic, separable embedding window + helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app import geo
from app.config import NODATA
from app.embeddings import EmbeddingWindow


@dataclass
class SimplePoint:
    lat: float
    lon: float
    cls: str


def make_window(h: int = 40, w: int = 40, epsg: int = 32610) -> EmbeddingWindow:
    """Left half = signature A, right half = signature B, with a NoData corner.

    Signatures differ across all 64 bands so a handful of points separate them.
    """
    arr = np.full((h, w, 64), 10, dtype=np.int8)
    half = w // 2
    arr[:, :half, :32] = 60          # class A: first 32 bands high
    arr[:, half:, 32:] = 60          # class B: last 32 bands high
    arr[0:2, 0:2, :] = NODATA        # masked corner
    return EmbeddingWindow(array=arr, x0=500005.0, y0=4185005.0, pix=10.0, epsg=epsg)


def make_point(win: EmbeddingWindow, row: int, col: int, cls: str) -> SimplePoint:
    """A point whose lon/lat lands on pixel (row, col) of ``win``."""
    x = win.x0 + col * win.pix
    y = win.y0 - row * win.pix
    lon, lat = geo.utm_to_lonlat(win.epsg, x, y)
    return SimplePoint(lat=lat, lon=lon, cls=cls)


@pytest.fixture
def window() -> EmbeddingWindow:
    return make_window()


@pytest.fixture
def points(window):
    a = [make_point(window, r, c, "A") for r, c in [(10, 5), (15, 8), (20, 3), (25, 10), (30, 6)]]
    b = [make_point(window, r, c, "B") for r, c in [(10, 30), (15, 33), (20, 28), (25, 35), (30, 31)]]
    return a + b
