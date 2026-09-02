import io

import numpy as np
from PIL import Image

from app import colorize


def test_parse_hex_variants():
    assert colorize.parse_hex("#ff0000") == (255, 0, 0)
    assert colorize.parse_hex("00ff00") == (0, 255, 0)
    assert colorize.parse_hex("#0f0") == (0, 255, 0)
    assert colorize.parse_hex("bogus") == (128, 128, 128)


def test_labels_to_png_colors_and_transparency():
    labels = np.array([[0, 1], [-1, 0]], dtype=np.int16)
    png = colorize.labels_to_png(labels, ["#ff0000", "#0000ff"])
    img = Image.open(io.BytesIO(png))
    assert img.size == (2, 2)  # (width, height)
    assert img.mode == "RGBA"
    px = img.load()
    assert px[0, 0] == (255, 0, 0, 255)   # class 0
    assert px[1, 0] == (0, 0, 255, 255)   # class 1
    assert px[0, 1][3] == 0               # -1 -> transparent
