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
uv run pytest -m live         # extra test that reads the REAL source (network, ~2 min)
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
| `INPROC_CACHE_SIZE` | `6` | Hot windows kept in RAM per process (~64 MB each). |
| `RATE_LIMIT` | `30/minute` | Per-IP limit on `/classify`. |
| `MAX_CLASSES` / `MAX_POINTS` | `12` / `500` | Request guard rails. |
| `DEFAULT_CLASSIFIER` | `rf` | `rf` (random forest) or `knn`. |
| `WEB_CONCURRENCY` | `2` | gunicorn workers (Docker). |

## Deploy (Google Cloud Run)

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/REPO/classifier
gcloud run deploy classifier \
  --image REGION-docker.pkg.dev/PROJECT/REPO/classifier \
  --memory 2Gi --cpu 2 --timeout 300 \
  --set-env-vars GCS_CACHE_BUCKET=YOUR_BUCKET

# optional: pre-fill the shared cache for the presets
GCS_CACHE_BUCKET=YOUR_BUCKET uv run python -m scripts.warm_cache
```

The Cloud Run service account needs read/write on the bucket. `/tmp` is a tmpfs
on Cloud Run (counts against memory), so always set `GCS_CACHE_BUCKET` in
production rather than relying on the disk fallback. Give it ≥ 2 GiB RAM (each
cached 10 km window is ~64 MB).

## Data & licensing

* AlphaEarth Foundations Satellite Embedding dataset — produced by Google and
  Google DeepMind, **CC-BY 4.0**, accessed via Source Cooperative.
* Basemap: Esri World Imagery (attribution required; no API key).
