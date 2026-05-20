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


# --- Coverage for #733: get_table_index NumPy / bisect refactor -------------


def _make_textline(x0, y0, x1, y1, text="x"):
    """Minimal stand-in for a PDFMiner LTTextLine sufficient for get_table_index."""

    class _TL:
        __slots__ = ("x0", "y0", "x1", "y1", "_objs", "_text")

        def __init__(self_, x0, y0, x1, y1, text):
            self_.x0, self_.y0, self_.x1, self_.y1 = x0, y0, x1, y1
            self_._objs = []
            self_._text = text

        def get_text(self_):
            return self_._text + "\n"

    return _TL(x0, y0, x1, y1, text)


def test_get_table_index_lazy_caches():
    """Table search caches are populated on first get_table_index call only."""
    from camelot.core import Table
    from camelot.utils import get_table_index

    rows = [(100.0, 90.0), (90.0, 80.0), (80.0, 70.0)]
    cols = [(0.0, 50.0), (50.0, 100.0)]
    table = Table(cols, rows)
    # caches empty initially
    assert table._rows_np_cache is None
    assert table._cols_np_cache is None
    assert table._neg_y_tops_list is None
    assert table._col_widths_list is None

    # Trigger the lookup
    textline = _make_textline(10.0, 84.0, 40.0, 86.0)
    indices, _err = get_table_index(table, textline, direction="horizontal")
    # exactly one (r, c, text) tuple
    assert len(indices) == 1
    r_idx, c_idx, text = indices[0]
    assert r_idx == 1  # row band (90, 80) contains y_mid=85
    assert c_idx == 0  # x_mid=25 is in column (0, 50)

    # caches now populated
    assert table._rows_np_cache is not None
    assert table._cols_np_cache is not None
    assert table._neg_y_tops_list == [-100.0, -90.0, -80.0]
    assert table._col_widths_list == [50.0, 50.0]
    assert table._rows_disjoint is True  # the disjoint fast-path triggered


def test_get_table_index_rows_overlap_fallback():
    """Overlapping rows force the linear-scan fallback (still bit-identical)."""
    from camelot.core import Table
    from camelot.utils import get_table_index

    # Deliberately overlap: y_bot of row 0 (85) < y_top of row 1 (90), so the
    # disjoint invariant rows[i].y_bot >= rows[i+1].y_top fails.
    rows = [(100.0, 85.0), (90.0, 75.0)]
    cols = [(0.0, 100.0)]
    table = Table(cols, rows)
    # Build caches and confirm we did NOT mark the rows disjoint
    table._build_search_caches()
    assert table._rows_disjoint is False

    # Textline mid-Y = 87.5 lies in both rows; the original code returns the
    # *first* matching row (index 0). The fallback path must do the same.
    textline = _make_textline(10.0, 86.0, 40.0, 89.0)
    indices, _err = get_table_index(table, textline, direction="horizontal")
    assert len(indices) == 1
    r_idx, c_idx, _text = indices[0]
    assert r_idx == 0
    assert c_idx == 0


def test_get_table_index_text_outside_any_row():
    """Textline whose mid-Y is outside every row band returns (empty, 1.0)."""
    from camelot.core import Table
    from camelot.utils import get_table_index

    rows = [(100.0, 90.0), (90.0, 80.0)]
    cols = [(0.0, 100.0)]
    table = Table(cols, rows)

    # mid-Y = 50 is below every row band
    textline = _make_textline(10.0, 49.0, 40.0, 51.0)
    indices, err = get_table_index(table, textline, direction="horizontal")
    assert indices == []
    assert err == 1.0
