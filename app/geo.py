"""Geometry helpers: UTM selection, fixed-size boxes, and point->pixel mapping.

Boxes are defined in the local UTM projection (metres) so a "10 km box" is
actually 10 km on the ground. AlphaEarth COGs are stored per UTM zone, which
makes UTM the natural working CRS and avoids reprojection for the common case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from pyproj import Transformer

from . import config


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=256)
def _to_utm(epsg: int) -> Transformer:
    return Transformer.from_crs(4326, epsg, always_xy=True)


@lru_cache(maxsize=256)
def _to_lonlat(epsg: int) -> Transformer:
    return Transformer.from_crs(epsg, 4326, always_xy=True)


def lonlat_to_utm(epsg: int, lon: float, lat: float) -> tuple[float, float]:
    return _to_utm(epsg).transform(lon, lat)


def utm_to_lonlat(epsg: int, x: float, y: float) -> tuple[float, float]:
    return _to_lonlat(epsg).transform(x, y)


@dataclass(frozen=True)
class Box:
    """A fixed-size square study area in a single UTM zone."""

    center_lon: float
    center_lat: float
    size_km: float
    epsg: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width_px(self) -> int:
        return int(round((self.xmax - self.xmin) / config.PIXEL_SIZE_M))

    @property
    def height_px(self) -> int:
        return int(round((self.ymax - self.ymin) / config.PIXEL_SIZE_M))

    def key(self) -> str:
        """Stable cache key (rounded to the metre)."""
        return (
            f"{self.epsg}_{int(round(self.xmin))}_{int(round(self.ymin))}"
            f"_{int(round(self.xmax))}_{int(round(self.ymax))}"
        )

    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) covering the four UTM corners.

        Used both for the aef-loader query and for the Leaflet image overlay.
        The UTM square is very slightly rotated relative to lon/lat, but at
        10 km the axis-aligned envelope is within a pixel or two.
        """
        corners = [
            utm_to_lonlat(self.epsg, x, y)
            for x in (self.xmin, self.xmax)
            for y in (self.ymin, self.ymax)
        ]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        return (min(lons), min(lats), max(lons), max(lats))


def build_box(
    lon: float,
    lat: float,
    size_km: float | None = None,
    snap_m: float | None = None,
) -> Box:
    """Build a fixed-size box centred on (lon, lat).

    The centre is snapped to a metre grid in UTM (``snap_m``) so that nearby
    free-entry coordinates resolve to the same cached window.
    """
    size_km = config.BOX_SIZE_KM if size_km is None else size_km
    snap_m = config.GRID_SNAP_M if snap_m is None else snap_m

    epsg = utm_epsg(lon, lat)
    cx, cy = lonlat_to_utm(epsg, lon, lat)
    if snap_m and snap_m > 0:
        cx = round(cx / snap_m) * snap_m
        cy = round(cy / snap_m) * snap_m

    half = size_km * 1000.0 / 2.0
    return Box(
        center_lon=lon,
        center_lat=lat,
        size_km=size_km,
        epsg=epsg,
        xmin=cx - half,
        ymin=cy - half,
        xmax=cx + half,
        ymax=cy + half,
    )


def point_rowcol(win, lon: float, lat: float) -> tuple[int, int] | None:
    """Map a lon/lat point to (row, col) in a north-up window, or None if outside.

    ``win`` is any object exposing ``epsg``, ``x0`` (easting of column 0),
    ``y0`` (northing of the top row), ``pix``, ``width`` and ``height``.
    """
    x, y = lonlat_to_utm(win.epsg, lon, lat)
    col = int(round((x - win.x0) / win.pix))
    row = int(round((win.y0 - y) / win.pix))  # y0 is the top (max northing)
    if 0 <= row < win.height and 0 <= col < win.width:
        return row, col
    return None
