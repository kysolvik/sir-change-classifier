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
    cache = EmbeddingCache(backend=_MemBackend(), inproc_size=4)
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
