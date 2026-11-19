"""Render an integer label array to a transparent RGBA PNG overlay."""
from __future__ import annotations

import colorsys
import io

import numpy as np
from PIL import Image


def parse_hex(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (128, 128, 128)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return (128, 128, 128)


def labels_to_png(labels: np.ndarray, colors: list[str]) -> bytes:
    """``labels`` is (H, W) int; class i uses ``colors[i]``; -1 is transparent."""
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask = labels == i
        if not mask.any():
            continue
        r, g, b = parse_hex(color)
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = 255

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def transition_palette(n: int) -> list[str]:
    """``n`` visually distinct colours (evenly-spaced hues at fixed S/V)."""
    out = []
    for i in range(n):
        h = i / n if n else 0.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out.append("#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255)))
    return out


def ids_to_png(ids: np.ndarray, color_by_id: dict[int, str]) -> bytes:
    """Render a sparse-id map (from ``transition_map``) to RGBA PNG.

    Ids not present in ``color_by_id`` (including the ``-1`` unchanged/nodata
    sentinel) are transparent.
    """
    h, w = ids.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for id_val, color in color_by_id.items():
        mask = ids == id_val
        if not mask.any():
            continue
        r, g, b = parse_hex(color)
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = 255

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --- raw imagery (Landsat composites) ---------------------------------------
# Band combinations students can view. A *fixed* reflectance stretch (not a
# per-image percentile one) keeps every year on the same scale, so a change in
# brightness between years is a real change on the ground.
IMAGERY_STYLES: dict[str, dict] = {
    "natural": {
        "label": "Natural colour",
        "bands": ("red", "green", "blue"),
        "max": 0.30,
    },
    "false": {
        "label": "False colour (plants are red)",
        "bands": ("nir08", "red", "green"),
        "max": 0.50,
    },
    "swir": {
        "label": "Shortwave infrared (bare ground & cities stand out)",
        "bands": ("swir22", "nir08", "red"),
        "max": 0.50,
    },
}
IMAGERY_GAMMA = 1.6  # brightens dark land without blowing out bright desert/ice


def bands_to_png(array: np.ndarray, band_names: list[str], style: str) -> bytes:
    """Render a float reflectance ``(H, W, bands)`` window as an RGBA PNG.

    NaN (cloud gaps / no data) pixels are transparent so the basemap shows through.
    """
    spec = IMAGERY_STYLES[style]
    rgb = np.stack([array[..., band_names.index(b)] for b in spec["bands"]], axis=-1)
    valid = np.all(np.isfinite(rgb), axis=-1)
    scaled = np.clip(np.nan_to_num(rgb) / spec["max"], 0, 1) ** (1 / IMAGERY_GAMMA)
    rgba = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = (scaled * 255).round().astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
