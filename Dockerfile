# Container for Cloud Run. rasterio/obstore/scikit-learn all ship manylinux
# wheels, so no system GDAL is required on top of the slim base.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    # /tmp is a tmpfs on Cloud Run (counts against memory); set GCS_CACHE_BUCKET
    # in production so cached windows live in GCS, not RAM.
    DISK_CACHE_DIR=/tmp/aef_cache \
    AEF_INDEX_DIR=/tmp/aef_index

WORKDIR /app

# Install dependencies first (cached layer) using the frozen lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY web ./web

# Cold embedding reads can take ~40 s; give gunicorn room so it doesn't kill
# the worker mid-request. Cloud Run injects $PORT (default 8080).
CMD uv run --no-dev gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:${PORT:-8080} \
    --workers ${WEB_CONCURRENCY:-2} \
    --timeout 180
