# sir-change-classifier

An interactive web tool that lets students experiment with remote-sensing
land-cover classification. A student picks a small study area, drops a handful
of labelled points for their own custom classes (water, forest, urban, …), and
the tool trains a simple classifier on **AlphaEarth Foundations satellite
embeddings** and paints the classification over the map. A year slider re-applies
the *same* trained model to any year from **2017–2025**.

No accounts, no rigid class list — inspired by Google Earth's experimental
classifier, built for a classroom.

## How it works

* **Embeddings.** [AlphaEarth Foundations](https://source.coop/tge-labs/aef) publishes
  64-band annual embeddings (10 m, 2017–2025) as int8 Cloud-Optimized GeoTIFFs on
  Source Cooperative. Each pixel is a unit vector in a space that is *consistent
  across years*, which is exactly what makes "train on one year, apply to another"
  valid and lets a few points per class separate cleanly.
* **Backend** (`app/`, FastAPI). Stateless: every `/classify` request carries its
  points and both years. It reads the embedding window for the box (via
  [`aef-loader`](https://pypi.org/project/aef-loader/)), samples the training
  points, fits a scikit-learn random forest (or KNN), applies it to the target
  year, and returns a colourised PNG overlay + stats. The only expensive step —
  reading a window — is cached and shared across all users.
* **Frontend** (`web/`, Leaflet + vanilla JS). Esri World Imagery basemap (free,
  no key), study-area picker, custom-class manager, click-to-label points, a
  year slider, and an opacity control. Projects persist to `localStorage` and can
  be exported/imported as JSON.

See `app/embeddings.py` for the notes learned about the data (multi-tile
mosaicking, the bottom-up y-axis, the int8 dequantization).

### Raw Landsat (second feature source, 1984–2025)

On the example places, students can switch **Features** to **Landsat bands**: six
raw surface-reflectance bands (blue, green, red, NIR, SWIR1, SWIR2) at 30 m,
from annual cloud-masked median composites (`app/imagery.py`) of USGS Collection 2
Level-2 scenes via Microsoft Planetary Computer. This reaches back to **1984**:

* **Sensors by year:** 2013+ uses Landsat 8/9 OLI. 1984–2012 uses Landsat 4/5 TM
  and Landsat 7 ETM+. After Landsat 7's scan-line failure (2003-05-31) its
  striped scenes only fill in behind Landsat 5, and 2012 (Landsat 7 only) takes
  twice the scenes so the stripes average out.
* **Comparing features:** *Compare feature types* (step 4) cross-validates the
  same classifier on every source using the same points and folds
  (`POST /api/evaluate`), overall and per class.
* **Imagery view:** step 3's *Background imagery* overlays the Landsat composite
  for any year (natural colour, false colour, SWIR) on the Esri basemap
  (`GET /api/imagery`), so students can label on the year they train on.

Compositing takes ~15–25 s per (place, year), so Landsat is **preset-only** by
default and the composites are **built offline** by the warm job (below). Adding
another source (e.g. raw Sentinel-2) means a reader in `app/cache.py` and an
entry in `config.FEATURE_SOURCES`; the API and UI are driven by that table.

## Run locally

```bash
uv sync
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000
```

Open the page, pick a preset (or type a lat/lon), add ~5 points per class, and
click **Train & classify**. The first look at a new area takes ~10–40 s while the
embeddings are fetched; after that it is cached and fast.

## Testing

```bash
uv run pytest                 # fast offline suite (data source is stubbed)
uv run pytest -m live         # tests that read the REAL sources (network, ~4 min)
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BOX_SIZE_KM` | `10` | Study-area size (100 km²). Shrink if performance demands. |
| `GRID_SNAP_M` | `1000` | Snap free-entry boxes to this grid so nearby entries share a cache entry. |
| `GCS_CACHE_BUCKET` | *(unset)* | Shared persistent cache bucket. If unset, falls back to `DISK_CACHE_DIR`. |
| `DISK_CACHE_DIR` | `/tmp/aef_cache` | Local persistent cache when no GCS bucket is set. |
| `AEF_INDEX_DIR` | `/tmp/aef_index` | Where aef-loader stores its tile index. |
| `COLD_FETCH_CONCURRENCY` | `4` | Max simultaneous cold reads from the source. |
| `INPROC_CACHE_SIZE` | `6` | RAM budget per process for hot windows, in 10 km AEF windows (~64 MB each). Landsat windows are ~2.7 MB. |
| `LANDSAT_ANY_AREA` | `0` | `1` offers Landsat everywhere, not just presets (each new box+year composites on request). |
| `LANDSAT_MAX_SCENES` / `LANDSAT_CLOUD_MAX` / `LANDSAT_RES_M` | `12` / `40` / `30` | Compositing knobs. |
| `LANDSAT_LOAD_THREADS` | `16` | Parallel scene reads per composite (~15 s vs ~90 s single-threaded). |
| `LANDSAT_HARMONIZE` | `0` | `1` applies the Roy et al. (2016) TM/ETM+→OLI adjustment (pre-collection coefficients; experimental). |
| `IMAGERY_RATE_LIMIT` | `240/minute` | Per-IP limit on `/api/imagery` (cheap once cached; students flip years fast). |
| `RATE_LIMIT` | `60/minute` | Per-IP limit on `/classify` and `/compare` (a classroom behind one NAT shares an IP). |
| `RATE_LIMIT_DEFAULT` | `300/minute` | Per-IP limit on every path, static files included. |
| `GLOBAL_CLASSIFY_PER_HOUR` | `150` | Classify+compare calls per worker process per hour, all IPs combined; beyond it the API returns 503. |
| `MAX_CLASSES` / `MAX_POINTS` | `12` / `500` | Request guard rails. |
| `DEFAULT_CLASSIFIER` | `rf` | `rf` (random forest) or `knn`. |
| `RF_N_ESTIMATORS` | `50` | Trees in the random forest; prediction time scales with it (~3.5 s per 10 km window on 2 vCPU). |
| `WEB_CONCURRENCY` | `2` | gunicorn workers (Docker). |

## Deploy (Google Cloud Run)

```bash
gcloud auth login && gcloud auth application-default login   # once
PROJECT=solvik-montero-sir ./scripts/deploy.sh                 # first deploy & every update
GCS_CACHE_BUCKET=solvik-montero-sir-aef-cache uv run python -m scripts.warm_cache   # once
# Landsat for every preset and year (~252 composites, ~1 h serially):
GCS_CACHE_BUCKET=solvik-montero-sir-aef-cache uv run python -m scripts.warm_cache --source landsat --all-years
```

Warm Landsat **before** students use it: an unwarmed (preset, year) composites
on the request path, which is slow (~15–25 s) and costs Cloud Run CPU. The total is
bounded, though: each of the 252 (preset, year) pairs is built at most once.

`scripts/deploy.sh` is idempotent. It enables the APIs, creates the GCS cache bucket and a
service account that can only touch that bucket, then builds and deploys from source. On the
first run it also sets up a billing budget and a request-flood alert. On redeploys, set
`SKIP_ALERTS=1` to skip that part. You can override `REGION`, `SERVICE`, `MAX_INSTANCES`,
`BUDGET_USD` and `ALERT_EMAIL`.

## Data & licensing

* AlphaEarth Foundations Satellite Embedding dataset — produced by Google and
  Google DeepMind, **CC-BY 4.0**, accessed via Source Cooperative.
* Landsat Collection 2 Level-2 — courtesy of the U.S. Geological Survey (public
  domain), accessed via Microsoft Planetary Computer.
* Optional harmonisation: Roy, D.P. et al. (2016), "Characterization of Landsat-7 to
  Landsat-8 reflective wavelength and normalized difference vegetation index
  continuity", *Remote Sensing of Environment* 185, 57–70.
* Basemap: Esri World Imagery (attribution required; no API key).
