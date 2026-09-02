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

    cache = EmbeddingCache(backend=InMemoryBackend(), inproc_size=4, reader=reader)
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
    cache._inproc.clear()  # force fall-through to the persistent backend
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
