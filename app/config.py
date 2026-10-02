"""Configuration constants and preset study areas.

Everything here is env-overridable so the same image can be tuned on Cloud Run
without a rebuild. The box size is deliberately a single knob (BOX_SIZE_KM) so we
can shrink it if performance demands.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --- Geometry --------------------------------------------------------------
PIXEL_SIZE_M = 10.0  # AlphaEarth native resolution
BOX_SIZE_KM = float(os.environ.get("BOX_SIZE_KM", "10"))  # 10 km => 100 km^2
# Free lat/lon entries are snapped to this grid (in the local UTM, metres) so
# nearby student entries collapse onto the same cached window. Snapping shifts
# the box centre by at most GRID_SNAP_M/2.
GRID_SNAP_M = float(os.environ.get("GRID_SNAP_M", "1000"))

# --- Years -----------------------------------------------------------------
MIN_YEAR = 2017
MAX_YEAR = 2025
DEFAULT_YEAR = 2025
YEARS = list(range(MIN_YEAR, MAX_YEAR + 1))

# --- Request limits (public-scale guard rails) -----------------------------
MAX_CLASSES = int(os.environ.get("MAX_CLASSES", "12"))
MAX_POINTS = int(os.environ.get("MAX_POINTS", "500"))
MIN_POINTS_TOTAL = 2  # need at least two labelled points across >=2 classes
# Per-IP limit on /classify and /compare. A whole classroom behind one school NAT
# shares an IP, so this must cover ~20 students, not one.
RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")
# Per-IP limit on every path (static files included) — a cheap flood guard.
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "300/minute")
# Hard cap on classify+compare calls per worker process per hour, regardless of
# IP. With Cloud Run max-instances this bounds total CPU spend even against a
# botnet rotating IPs (total = cap x workers x max instances).
GLOBAL_CLASSIFY_PER_HOUR = int(os.environ.get("GLOBAL_CLASSIFY_PER_HOUR", "150"))

# --- Caching ---------------------------------------------------------------
# Number of (box, year) embedding windows kept hot in process RAM. Each 10 km
# window is ~64 MB, so keep this modest.
INPROC_CACHE_SIZE = int(os.environ.get("INPROC_CACHE_SIZE", "6"))
# Bound simultaneous cold reads from the source (prevents a stampede on a class
# all hitting a new area at once).
COLD_FETCH_CONCURRENCY = int(os.environ.get("COLD_FETCH_CONCURRENCY", "4"))
# Persistent shared cache. If GCS_CACHE_BUCKET is set we use it; otherwise we
# fall back to a local disk directory (handy for dev and single-instance runs).
GCS_CACHE_BUCKET = os.environ.get("GCS_CACHE_BUCKET", "").strip()
GCS_CACHE_PREFIX = os.environ.get("GCS_CACHE_PREFIX", "windows").strip("/")
DISK_CACHE_DIR = os.environ.get("DISK_CACHE_DIR", "/tmp/aef_cache")
# Where aef-loader stores its downloaded tile index.
AEF_INDEX_DIR = os.environ.get("AEF_INDEX_DIR", "/tmp/aef_index")

# --- Classifier ------------------------------------------------------------
DEFAULT_CLASSIFIER = os.environ.get("DEFAULT_CLASSIFIER", "rf")  # "rf" | "knn"
# Prediction cost scales with tree count: on 2 vCPU a full 10 km window takes
# ~3.5 s at 50 trees vs ~9.6 s at 150. With a few dozen training points the
# extra trees add little.
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "50"))
KNN_NEIGHBORS = int(os.environ.get("KNN_NEIGHBORS", "5"))
NODATA = -128  # AlphaEarth int8 NoData sentinel


@dataclass(frozen=True)
class Preset:
    name: str
    lat: float
    lon: float
    blurb: str


# A handful of visually legible study areas spanning contrasting land cover.
PRESETS: list[Preset] = [
    Preset("San Francisco Bay, USA", 37.780, -122.440, "City, water, parks, hills"),
    Preset("Central Valley, California", 36.700, -120.000, "Irrigated agriculture & fallow"),
    Preset("Nile Delta, Egypt", 30.700, 31.100, "Farmland, desert, water, towns"),
    Preset("Rondonia, Brazil", -10.200, -62.700, "Rainforest & deforestation edges"),
    Preset("Dubai coast, UAE", 25.150, 55.200, "Urban, desert, sea"),
    Preset("Vatnajokull margin, Iceland", 64.200, -16.800, "Ice, water, bare rock, tundra"),
]
