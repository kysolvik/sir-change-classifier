"""Reading AlphaEarth embedding windows via aef-loader.

Returns a north-up ``(H, W, 64)`` int8 array for a :class:`~app.geo.Box` and a
year, together with the affine origin needed to map points to pixels.

Notes learned from the data (see the spike):
  * ``index.query`` returns a list of ``AEFTileInfo`` (each carries ``crs_epsg``
    and an ``s3://`` path); ``open_tiles_by_zone`` mosaics same-zone tiles into a
    single lazy (dask-backed) dataset per UTM zone.
  * The ``embeddings`` variable is ``(time, band, y, x)`` int8; ``y`` is
    *ascending* (row 0 = south), so we flip to north-up.
  * A box that straddles two UTM zones yields multiple zone groups; for v1 we
    keep only the centre zone (a box exactly on a 6-degree seam is clipped).
"""
from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass

import numpy as np

from . import config, geo


class NoDataError(RuntimeError):
    """Raised when no embedding coverage exists for the requested box/year."""


@dataclass
class EmbeddingWindow:
    array: np.ndarray  # (H, W, 64) int8, north-up
    x0: float  # easting of column 0 (leftmost pixel centre)
    y0: float  # northing of row 0 (top pixel centre, max northing)
    pix: float
    epsg: int

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


# --- aef-loader index singleton -------------------------------------------
_index = None
_index_lock = asyncio.Lock()
_read_semaphore: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _read_semaphore
    if _read_semaphore is None:
        _read_semaphore = asyncio.Semaphore(config.COLD_FETCH_CONCURRENCY)
    return _read_semaphore


async def get_index():
    """Download + load the tile index once per process."""
    global _index
    if _index is None:
        async with _index_lock:
            if _index is None:
                from aef_loader import AEFIndex, DataSource

                idx = AEFIndex(
                    source=DataSource.SOURCE_COOP,
                    cache_dir=pathlib.Path(config.AEF_INDEX_DIR),
                )
                await idx.download()
                idx.load()
                _index = idx
    return _index


def _materialize(zone_ds, box: geo.Box) -> EmbeddingWindow:
    """Window a lazy zone dataset to the box and pull the pixels (blocking)."""
    sub = zone_ds.sel(x=slice(box.xmin, box.xmax), y=slice(box.ymin, box.ymax))
    if sub.sizes.get("x", 0) == 0 or sub.sizes.get("y", 0) == 0:
        raise NoDataError("box falls outside embedding coverage")

    # (band, y, x) -> (y, x, band); .values triggers the dask compute.
    vals = sub["embeddings"].isel(time=0).values
    arr = np.transpose(vals, (1, 2, 0))

    xc = np.asarray(sub["x"].values)
    yc = np.asarray(sub["y"].values)
    if yc[0] < yc[-1]:  # ascending (south->north): flip to north-up
        arr = arr[::-1, :, :]
    x0 = float(xc.min())
    y0 = float(yc.max())
    return EmbeddingWindow(
        array=np.ascontiguousarray(arr),
        x0=x0,
        y0=y0,
        pix=config.PIXEL_SIZE_M,
        epsg=box.epsg,
    )


async def read_window(box: geo.Box, year: int) -> EmbeddingWindow:
    """Read the embedding window for ``box`` and ``year`` from the source.

    Bounded by a semaphore so a burst of cold reads can't overwhelm the
    process or the source endpoint.
    """
    from aef_loader import VirtualTiffReader

    index = await get_index()
    tiles = await index.query(bbox=box.lonlat_bounds(), years=(year, year))
    tiles = [t for t in tiles if t.crs_epsg == box.epsg]  # keep the centre zone
    if not tiles:
        raise NoDataError(f"no embedding tiles for {box.epsg} in {year}")

    async with _semaphore():
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)
            zone_ds = _select_zone(tree, box.epsg)
            return await asyncio.to_thread(_materialize, zone_ds, box)


def _select_zone(tree, epsg: int):
    """Pick the DataTree zone group whose CRS matches ``epsg``."""
    children = list(tree.children)
    for name in children:
        ds = tree[name].ds
        try:
            if int(np.asarray(ds["spatial_ref"].values)) == epsg:
                return ds
        except Exception:
            continue
    # Fall back to the only/first zone.
    return tree[children[0]].ds
