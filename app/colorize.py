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
