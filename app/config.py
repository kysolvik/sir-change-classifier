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

# --- Years (AlphaEarth; each feature source has its own range below) --------
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
# Per-IP limit on imagery views. Cheap once cached, and students flip through
# years quickly, so this is looser than RATE_LIMIT.
IMAGERY_RATE_LIMIT = os.environ.get("IMAGERY_RATE_LIMIT", "240/minute")
# Per-IP limit on every path (static files included) — a cheap flood guard.
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "300/minute")
# Hard cap on classify+compare calls per worker process per hour, regardless of
# IP. With Cloud Run max-instances this bounds total CPU spend even against a
# botnet rotating IPs (total = cap x workers x max instances).
GLOBAL_CLASSIFY_PER_HOUR = int(os.environ.get("GLOBAL_CLASSIFY_PER_HOUR", "150"))

# --- Caching ---------------------------------------------------------------
# Process-RAM budget for hot windows, in units of one 10 km AEF window (~64 MB).
# Budgeted by bytes, not count, so browsing many small Landsat composites
# (~2.7 MB each) doesn't evict the large embedding windows.
INPROC_CACHE_SIZE = int(os.environ.get("INPROC_CACHE_SIZE", "6"))
INPROC_CACHE_BYTES = INPROC_CACHE_SIZE * 64 * 1024 * 1024
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

# --- Raw imagery features (Landsat) ----------------------------------------
# A second feature source: raw Landsat surface-reflectance bands instead of AEF
# embeddings, reaching back to Landsat 5 (1984). Compositing is slow (~50 s per
# box+year), so by default raw mode is offered on presets only and the composites
# are pre-warmed offline (scripts/warm_cache.py). LANDSAT_ANY_AREA=1 lifts the
# preset gate (handy locally; each new box+year then composites on request).
LANDSAT_ANY_AREA = os.environ.get("LANDSAT_ANY_AREA", "0") == "1"
PC_STAC_URL = os.environ.get(
    "PC_STAC_URL", "https://planetarycomputer.microsoft.com/api/stac/v1"
)
LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_MIN_YEAR = 1984
LANDSAT_MAX_YEAR = 2025
# Planetary Computer asset keys are the same for TM, ETM+ and OLI.
LANDSAT_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22"]
LANDSAT_RES_M = float(os.environ.get("LANDSAT_RES_M", "30"))
LANDSAT_CLOUD_MAX = int(os.environ.get("LANDSAT_CLOUD_MAX", "40"))
LANDSAT_MAX_SCENES = int(os.environ.get("LANDSAT_MAX_SCENES", "12"))
# Threads odc-stac uses to read scene windows. Loading is network-latency bound
# (many small COG range requests), so threads help far beyond the CPU count:
# measured ~12-20 s per composite with 16 vs ~70-105 s single-threaded.
LANDSAT_LOAD_THREADS = int(os.environ.get("LANDSAT_LOAD_THREADS", "16"))
# Landsat 7's scan-line corrector failed on 2003-05-31, leaving ~22% of every
# later scene as no-data stripes. Its scenes are used only to fill in behind
# Landsat 5 / 8; a year that has nothing else (2012) takes this many more scenes
# so the stripes average out.
LANDSAT7_SLC_OFF = "2003-05-31"
LANDSAT_SLC_OFF_SCENE_FACTOR = 2
# Landsat C2 L2 surface-reflectance scaling, and QA_PIXEL bits to drop:
# fill(0), dilated cloud(1), cirrus(2), cloud(3), cloud shadow(4).
LANDSAT_SR_SCALE = 2.75e-5
LANDSAT_SR_OFFSET = -0.2
LANDSAT_QA_MASK = 0b11111
# Optional cross-sensor adjustment, OFF by default. Roy et al. (2016, RSE 185)
# Table 2 OLS coefficients map ETM+ onto OLI (OLI = intercept + slope * ETM+),
# but they were fitted on *pre-collection* data and have not been re-derived for
# Collection 2, which this app reads. The current consensus (Earth Engine FAQ,
# citing the Landsat Science Team) is that C2 surface reflectance needs no
# cross-sensor correction for land-cover mapping or change detection. Kept as an
# opt-in experiment; applying ETM+ coefficients to TM is a further approximation.
LANDSAT_HARMONIZE = os.environ.get("LANDSAT_HARMONIZE", "0") == "1"
LANDSAT_ETM_TO_OLI = {  # band: (slope, intercept)
    "blue": (0.8474, 0.0003),
    "green": (0.8483, 0.0088),
    "red": (0.9047, 0.0061),
    "nir08": (0.8462, 0.0412),
    "swir16": (0.8937, 0.0254),
    "swir22": (0.9071, 0.0172),
}
# Bump when compositing changes so stale cached composites are not reused. The
# harmonisation setting is part of the key too (see cache.EmbeddingCache.key).
LANDSAT_CACHE_VERSION = "v3"


@dataclass(frozen=True)
class FeatureSource:
    """One kind of per-pixel features a model can be trained on."""

    id: str
    label: str
    min_year: int
    max_year: int
    res_m: float
    preset_only: bool  # only offered on the preset study areas
    imagery: bool  # can be rendered as a picture (raw bands, not embeddings)

    @property
    def years(self) -> list[int]:
        return list(range(self.min_year, self.max_year + 1))


# Registry of feature sources. Adding one (e.g. raw Sentinel-2) means a reader in
# cache.py plus an entry here; the API and frontend are driven by this table.
FEATURE_SOURCES: dict[str, FeatureSource] = {
    "aef": FeatureSource(
        "aef", "AlphaEarth embeddings", MIN_YEAR, MAX_YEAR, PIXEL_SIZE_M,
        preset_only=False, imagery=False,
    ),
    "landsat": FeatureSource(
        "landsat", "Landsat bands", LANDSAT_MIN_YEAR, LANDSAT_MAX_YEAR, LANDSAT_RES_M,
        preset_only=not LANDSAT_ANY_AREA, imagery=True,
    ),
}
DEFAULT_FEATURE_SOURCE = "aef"


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
