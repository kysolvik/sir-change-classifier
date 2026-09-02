"""Two-tier cache for embedding windows.

Tier 1: a small per-process LRU (hot windows in RAM).
Tier 2: a shared persistent store (a GCS bucket in production, or a local disk
        directory for dev / single instance).

Reading a 10 km window from the source costs ~10 s, so the whole point is to pay
that once and share it. An in-flight map additionally de-duplicates concurrent
requests for the same window *within* a process, so a class all opening the same
preset at once triggers a single read, not thirty.
"""
from __future__ import annotations

import asyncio
import io
import os
from collections import OrderedDict

import numpy as np

from . import config, embeddings
from .embeddings import EmbeddingWindow
from .geo import Box


# --- (de)serialisation -----------------------------------------------------
def encode(win: EmbeddingWindow) -> bytes:
    buf = io.BytesIO()
    meta = np.array([win.x0, win.y0, win.pix, float(win.epsg)], dtype=np.float64)
    np.savez_compressed(buf, array=win.array, meta=meta)
    return buf.getvalue()


def decode(data: bytes) -> EmbeddingWindow:
    with np.load(io.BytesIO(data)) as npz:
        array = npz["array"]
        x0, y0, pix, epsg = npz["meta"].tolist()
    return EmbeddingWindow(array=array, x0=x0, y0=y0, pix=pix, epsg=int(epsg))


# --- persistent backends ---------------------------------------------------
class DiskBackend:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.npz")

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
        return None

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)  # atomic


class GCSBackend:
    def __init__(self, bucket: str, prefix: str):
        from google.cloud import storage

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix

    def _blob(self, key: str):
        return self._bucket.blob(f"{self.prefix}/{key}.npz")

    def get(self, key: str) -> bytes | None:
        blob = self._blob(key)
        if blob.exists():
            return blob.download_as_bytes()
        return None

    def put(self, key: str, data: bytes) -> None:
        self._blob(key).upload_from_string(data, content_type="application/octet-stream")


def _build_backend():
    if config.GCS_CACHE_BUCKET:
        return GCSBackend(config.GCS_CACHE_BUCKET, config.GCS_CACHE_PREFIX)
    return DiskBackend(config.DISK_CACHE_DIR)


# --- the cache ------------------------------------------------------------
class EmbeddingCache:
    def __init__(self, backend=None, inproc_size: int | None = None, reader=None):
        self.backend = backend if backend is not None else _build_backend()
        self.inproc_size = inproc_size if inproc_size is not None else config.INPROC_CACHE_SIZE
        self._reader = reader or embeddings.read_window
        self._inproc: OrderedDict[str, EmbeddingWindow] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self.source_reads = 0  # cold reads that actually hit the source

    def _hot_get(self, key: str) -> EmbeddingWindow | None:
        win = self._inproc.get(key)
        if win is not None:
            self._inproc.move_to_end(key)
        return win

    def _hot_put(self, key: str, win: EmbeddingWindow) -> None:
        self._inproc[key] = win
        self._inproc.move_to_end(key)
        while len(self._inproc) > self.inproc_size:
            self._inproc.popitem(last=False)

    async def get_or_read(self, box: Box, year: int) -> EmbeddingWindow:
        key = f"{box.key()}_{year}"

        hot = self._hot_get(key)
        if hot is not None:
            return hot

        inflight = self._inflight.get(key)
        if inflight is not None:
            return await inflight

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[key] = fut
        try:
            win = await self._load(key, box, year)
            self._hot_put(key, win)
            fut.set_result(win)
            return win
        except Exception as exc:  # propagate to any awaiters
            fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)

    async def _load(self, key: str, box: Box, year: int) -> EmbeddingWindow:
        data = await asyncio.to_thread(self.backend.get, key)
        if data is not None:
            return decode(data)
        # Cold: read from the source, then persist for everyone else.
        self.source_reads += 1
        win = await self._reader(box, year)
        try:
            await asyncio.to_thread(self.backend.put, key, encode(win))
        except Exception:
            pass  # a cache write failure must not fail the request
        return win


cache = EmbeddingCache()
