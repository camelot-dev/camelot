"""#40: in-memory render (skip the PNG round-trip) must match the file path."""

import os

import numpy as np

import camelot
from camelot.backends import ImageConversionBackend
from camelot.image_processing import adaptive_threshold
from camelot.utils import build_file_path_in_temp_dir


def test_to_array_matches_convert_imread(testdir):
    import cv2

    pdf = os.path.join(testdir, "foo.pdf")
    icb = ImageConversionBackend()
    arr = icb.to_array(pdf, page=1)
    png = build_file_path_in_temp_dir("inmem_ref", ".png")
    icb.convert(pdf, png, page=1)
    ref = cv2.imread(png)
    assert arr.shape == ref.shape  # same H x W x 3 (BGR)
    # PNG is lossless, so the in-memory render is the same pixels (allow a
    # 1-LSB tolerance against any encoder rounding).
    assert np.allclose(arr, ref, atol=1)


def test_adaptive_threshold_accepts_array(testdir):
    pdf = os.path.join(testdir, "foo.pdf")
    arr = ImageConversionBackend().to_array(pdf, page=1)
    img, threshold = adaptive_threshold(arr)
    assert img.shape[:2] == threshold.shape


def test_lattice_inmemory_matches_pngpath(testdir):
    # End-to-end: the lattice raster engine (now in-memory) gives the same
    # table as before. foo.pdf is the canonical ruled fixture.
    pdf = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(pdf, flavor="lattice", engine="raster")
    assert len(tables) == 1
    assert tables[0].shape == (7, 7)
