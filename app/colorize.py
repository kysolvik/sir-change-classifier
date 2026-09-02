"""Render an integer label array to a transparent RGBA PNG overlay."""
from __future__ import annotations

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
