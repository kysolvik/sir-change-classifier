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


def test_transition_palette_distinct():
    colors = colorize.transition_palette(6)
    assert len(colors) == 6
    assert len(set(colors)) == 6                      # all distinct
    assert all(len(c) == 7 and c[0] == "#" for c in colors)
    assert all(colorize.parse_hex(c) != (128, 128, 128) for c in colors)  # all parse
    assert colorize.transition_palette(0) == []


def test_ids_to_png_maps_ids_and_is_transparent_off_map():
    # ids: 5 and 7 coloured; -1 and an unmapped id stay transparent.
    ids = np.array([[5, 7], [-1, 9]], dtype=np.int32)
    png = colorize.ids_to_png(ids, {5: "#ff0000", 7: "#0000ff"})
    img = Image.open(io.BytesIO(png))
    assert img.size == (2, 2)
    px = img.load()
    assert px[0, 0] == (255, 0, 0, 255)   # id 5
    assert px[1, 0] == (0, 0, 255, 255)   # id 7
    assert px[0, 1][3] == 0               # -1 -> transparent
    assert px[1, 1][3] == 0               # unmapped id 9 -> transparent
