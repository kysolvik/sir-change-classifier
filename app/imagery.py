"""Build a cloud-free annual Landsat surface-reflectance composite for a box + year.

The raw-band alternative to AEF embeddings, reaching back to 1984. Compositing
reads many scenes (~50 s per box+year), so it is meant to run in the **offline**
warm job (scripts/warm_cache.py); production then reads the result from the
shared cache. Returns an :class:`~app.embeddings.EmbeddingWindow` (float32
``(H, W, n_bands)``, NaN = masked) so classify/colorize are unchanged.

Sensor policy (see :func:`select_scenes`):
  * 2013+          Landsat 8/9 OLI only (Landsat 7 is striped by then).
  * 1984-2012      Landsat 4/5 TM and Landsat 7 ETM+. After Landsat 7's
                   scan-line-corrector failure (2003-05-31) its scenes rank
                   behind Landsat 5's, so they only fill in. 2012 has only
                   Landsat 7, so it takes extra scenes to fill the stripes.
Collection 2 surface reflectance is used as-is across sensors (the consensus for
C2); an opt-in Roy et al. (2016) TM/ETM+ -> OLI adjustment is available via
LANDSAT_HARMONIZE=1, though its coefficients predate Collection 2.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import warnings

import numpy as np

from . import config, embeddings, geo
from .embeddings import EmbeddingWindow, NoDataError

OLI_PLATFORMS = ("landsat-8", "landsat-9")
LEGACY_PLATFORMS = ("landsat-4", "landsat-5", "landsat-7")
OLI_FIRST_YEAR = 2013


async def read_composite(box: geo.Box, year: int) -> EmbeddingWindow:
    """Async wrapper (matches ``embeddings.read_window``): builds in a thread,
    bounded by the shared cold-fetch semaphore."""
    async with embeddings._semaphore():
        return await asyncio.to_thread(_build_composite, box, year)


def platforms_for(year: int) -> tuple[str, ...]:
    return OLI_PLATFORMS if year >= OLI_FIRST_YEAR else LEGACY_PLATFORMS


def _slc_off(item) -> bool:
    if item.properties.get("platform") != "landsat-7":
        return False
    return item.datetime.date() >= dt.date.fromisoformat(config.LANDSAT7_SLC_OFF)


def select_scenes(items: list) -> list:
    """Pick the scenes to composite: least cloudy first, striped Landsat 7 last.

    Takes ``LANDSAT_MAX_SCENES``; if the pick is striped Landsat 7 only, takes
    ``LANDSAT_SLC_OFF_SCENE_FACTOR`` times as many so the gaps fill in.
    """
    ranked = sorted(
        items, key=lambda it: (_slc_off(it), it.properties.get("eo:cloud_cover", 100))
    )
    n = config.LANDSAT_MAX_SCENES
    if ranked and all(_slc_off(it) for it in ranked[:n]):
        n *= config.LANDSAT_SLC_OFF_SCENE_FACTOR
    return ranked[:n]


def to_reflectance(dn: np.ndarray, band: str, legacy: bool) -> np.ndarray:
    """Scale C2 L2 digital numbers to reflectance (optionally adjust TM/ETM+ to OLI)."""
    sr = dn.astype("float32") * config.LANDSAT_SR_SCALE + config.LANDSAT_SR_OFFSET
    if legacy and config.LANDSAT_HARMONIZE:
        slope, intercept = config.LANDSAT_ETM_TO_OLI[band]
        sr = sr * slope + intercept
    return sr


def _build_composite(box: geo.Box, year: int) -> EmbeddingWindow:
    import odc.stac
    import planetary_computer
    import pystac_client
    from odc.geo.geobox import GeoBox

    catalog = pystac_client.Client.open(
        config.PC_STAC_URL, modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(
        collections=[config.LANDSAT_COLLECTION],
        bbox=box.lonlat_bounds(),
        datetime=f"{year}-01-01/{year}-12-31",
        query={
            "eo:cloud_cover": {"lt": config.LANDSAT_CLOUD_MAX},
            "platform": {"in": list(platforms_for(year))},
        },
    )
    items = select_scenes(list(search.items()))
    if not items:
        raise NoDataError(f"No usable Landsat scenes here for {year}.")
    legacy = year < OLI_FIRST_YEAR

    geobox = GeoBox.from_bbox(
        (box.xmin, box.ymin, box.xmax, box.ymax),
        crs=f"EPSG:{box.epsg}",
        resolution=config.LANDSAT_RES_M,
        tight=True,
    )
    ds = odc.stac.load(
        items,
        bands=config.LANDSAT_BANDS + ["qa_pixel"],
        geobox=geobox,
        groupby="solar_day",
        pool=config.LANDSAT_LOAD_THREADS,  # parallel reads: the big speed lever
    )
    ds.load()

    # Drop cloud/shadow/cirrus/fill pixels, scale to reflectance, median over time.
    invalid = (ds["qa_pixel"].values.astype("uint16") & config.LANDSAT_QA_MASK) != 0
    bands = []
    for b in config.LANDSAT_BANDS:
        sr = to_reflectance(ds[b].values, b, legacy)
        sr[invalid] = np.nan
        with warnings.catch_warnings():  # an always-cloudy pixel is an all-NaN slice
            warnings.simplefilter("ignore", RuntimeWarning)
            bands.append(np.clip(np.nanmedian(sr, axis=0), 0, 1))
    arr = np.stack(bands, axis=-1).astype("float32")  # (H, W, n_bands); NaN = masked
    if not np.isfinite(arr).any():
        raise NoDataError(f"Landsat scenes for {year} are all cloud here.")

    xc = np.asarray(ds["x"].values)
    yc = np.asarray(ds["y"].values)
    if yc[0] < yc[-1]:  # ascending -> flip to north-up (odc is normally descending)
        arr = arr[::-1, :, :]
    return EmbeddingWindow(
        array=np.ascontiguousarray(arr),
        x0=float(xc.min()),
        y0=float(yc.max()),
        pix=config.LANDSAT_RES_M,
        epsg=box.epsg,
    )

