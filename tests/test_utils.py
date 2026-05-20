"""Test to check intersection logic when no intersection area returned."""

import os

import playa.miner as pm
from playa.miner import LAParams
from playa.miner import LTTextBoxHorizontal

from camelot.utils import bbox_intersection_area


def get_text_from_pdf(filename):
    """Method to extract text object from pdf."""
    for layout in pm.extract(filename, LAParams()):
        for element in layout:
            if isinstance(element, LTTextBoxHorizontal):
                return element


def test_bbox_intersection_text(testdir):
    """
    Test to check area of intersection between both boxes when no intersection area returned.
    """
    filename1 = os.path.join(testdir, "foo.pdf")
    pdftextelement1 = get_text_from_pdf(filename1)
    filename2 = os.path.join(testdir, "tabula/12s0324.pdf")
    pdftextelement2 = get_text_from_pdf(filename2)

    assert bbox_intersection_area(pdftextelement1, pdftextelement2) == 0.0


# --- Coverage for camelot.utils helpers refactored in #718 -------------------


def test_random_string_length_and_alphabet():
    """random_string returns a string of the requested length over the digit + letter alphabet."""
    import string

    from camelot.utils import random_string

    alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase

    s = random_string(0)
    assert s == ""

    s = random_string(32)
    assert isinstance(s, str)
    assert len(s) == 32
    assert all(ch in alphabet for ch in s)


def test_text_in_bbox_filters_and_discards_overlaps():
    """text_in_bbox keeps boxes whose centre is inside, then drops 80%-contained shorter siblings."""
    from camelot.utils import text_in_bbox

    class _Box:
        """Minimal stand-in for a PDFMiner-style text object."""

        def __init__(self, x0, y0, x1, y1, txt=""):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
            self._txt = txt

        def get_text(self):
            return self._txt

    # Three non-overlapping boxes, all inside the query bbox: all kept.
    a = _Box(0, 0, 10, 5)
    b = _Box(20, 0, 30, 5)
    c = _Box(40, 0, 50, 5)
    assert set(text_in_bbox((0, 0, 60, 10), [a, b, c])) == {a, b, c}

    # A box whose centre is outside the query bbox is dropped by the filter.
    d_outside = _Box(100, 100, 110, 110)
    assert d_outside not in text_in_bbox((0, 0, 60, 10), [a, d_outside])

    # A shorter box ~fully contained in a longer one is discarded.
    big = _Box(0, 0, 100, 5, "BIGGER LINE")
    small = _Box(10, 1, 14, 4, "x")  # >80% inside big, but shorter
    out = text_in_bbox((0, 0, 200, 10), [big, small])
    assert big in out
    assert small not in out

    # Empty input — empty output, no crash.
    assert text_in_bbox((0, 0, 100, 100), []) == []
