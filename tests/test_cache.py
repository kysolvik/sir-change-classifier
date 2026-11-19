import asyncio

import pytest

from app import geo
from app.cache import EmbeddingCache
from tests.conftest import make_window


class InMemoryBackend:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, data):
        self.store[key] = data


def make_cache(delay=0.0):
    calls = {"n": 0}

    async def reader(box, year):
        calls["n"] += 1
        if delay:
            await asyncio.sleep(delay)
        return make_window()

    cache = EmbeddingCache(backend=InMemoryBackend(), inproc_bytes=1 << 30, reader=reader)
    return cache, calls


@pytest.mark.asyncio
async def test_hot_cache_avoids_second_read():
    cache, calls = make_cache()
    box = geo.build_box(-122.44, 37.78)
    w1 = await cache.get_or_read(box, 2025)
    w2 = await cache.get_or_read(box, 2025)
    assert calls["n"] == 1
    assert cache.source_reads == 1
    assert w1 is w2  # served from hot tier


@pytest.mark.asyncio
async def test_persistent_tier_used_when_hot_evicted():
    cache, calls = make_cache()
    box = geo.build_box(-122.44, 37.78)
    await cache.get_or_read(box, 2025)
    cache.clear_hot()  # force fall-through to the persistent backend
    await cache.get_or_read(box, 2025)
    assert calls["n"] == 1  # decoded from backend, no new source read


@pytest.mark.asyncio
async def test_distinct_years_read_separately():
    cache, calls = make_cache()
    box = geo.build_box(-122.44, 37.78)
    await cache.get_or_read(box, 2025)
    await cache.get_or_read(box, 2017)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_concurrent_requests_dedupe_to_one_read():
    cache, calls = make_cache(delay=0.1)
    box = geo.build_box(-122.44, 37.78)
    results = await asyncio.gather(*[cache.get_or_read(box, 2025) for _ in range(10)])
    assert calls["n"] == 1  # in-flight de-duplication
    assert all(r is results[0] for r in results)


def test_aef_key_is_unchanged_and_sources_dont_collide():
    box = geo.build_box(-122.44, 37.78)
    # Production's warmed AEF windows use this exact key; it must not move.
    assert EmbeddingCache.key(box, 2020) == f"{box.key()}_2020"
    assert EmbeddingCache.key(box, 2020, "landsat") != EmbeddingCache.key(box, 2020)


def test_landsat_key_tracks_harmonisation(monkeypatch):
    from app import config

    box = geo.build_box(-122.44, 37.78)
    plain = EmbeddingCache.key(box, 1990, "landsat")
    monkeypatch.setattr(config, "LANDSAT_HARMONIZE", True)
    assert EmbeddingCache.key(box, 1990, "landsat") != plain


@pytest.mark.asyncio
async def test_inproc_is_budgeted_by_bytes():
    win = make_window()  # 40*40*64 int8 = 102,400 bytes
    cache = EmbeddingCache(backend=InMemoryBackend(), inproc_bytes=250_000)

    async def reader(box, year):
        return make_window()

    cache._reader = reader
    box = geo.build_box(-122.44, 37.78)
    for year in (2017, 2018, 2019):
        await cache.get_or_read(box, year)
    assert len(cache._inproc) == 2 and cache._inproc_used == 2 * win.array.nbytes
    assert list(cache._inproc) == [EmbeddingCache.key(box, y) for y in (2018, 2019)]
