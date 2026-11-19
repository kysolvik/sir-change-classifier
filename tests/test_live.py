"""Live integration test against the real AlphaEarth source (network required).

Run explicitly with:  uv run pytest -m live -v
Skipped by default in the offline suite (uv run pytest -m 'not live').
"""
import numpy as np
import pytest

from app import geo
from app.cache import EmbeddingCache

pytestmark = pytest.mark.live


class _MemBackend:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, data):
        self.store[key] = data


@pytest.mark.asyncio
async def test_real_window_read_and_cache():
    cache = EmbeddingCache(backend=_MemBackend(), inproc_bytes=1 << 30)
    box = geo.build_box(-122.42, 37.775)  # San Francisco

    win = await cache.get_or_read(box, 2025)
    assert win.array.shape == (box.height_px, box.width_px, 64)
    assert win.array.dtype == np.int8
    # A real land window should have mostly valid (non-NoData) pixels.
    valid = np.mean(np.any(win.array != -128, axis=2))
    assert valid > 0.5

    # Second identical read is served from cache: no new source read.
    await cache.get_or_read(box, 2025)
    assert cache.source_reads == 1

    # Another year is available and reads independently.
    await cache.get_or_read(box, 2017)
    assert cache.source_reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("year", [2020, 1990])  # Landsat 8 OLI, and Landsat 5 TM
async def test_real_landsat_composite(year):
    """Build a real Landsat composite (slow, ~40-60 s each) and sanity-check it."""
    from app import config, imagery

    box = geo.build_box(-62.700, -10.200)  # Rondonia
    win = await imagery.read_composite(box, year)

    assert win.array.shape[-1] == len(config.LANDSAT_BANDS)
    assert win.array.dtype == np.float32
    assert win.pix == config.LANDSAT_RES_M
    # Mostly-valid (non-NaN) reflectance in [0, 1].
    valid = np.isfinite(win.array[..., 0])
    assert np.mean(valid) > 0.5
    assert np.nanmax(win.array) <= 1.0 and np.nanmin(win.array) >= 0.0
