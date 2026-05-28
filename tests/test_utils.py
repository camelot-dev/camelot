"""Test to check intersection logic when no intersection area returned."""

import os

import playa.miner as pm
import pytest
from playa.miner import LAParams
from playa.miner import LTTextBoxHorizontal

from camelot.utils import bbox_from_str
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
    """text_in_bbox keeps boxes whose centre is inside, then drops 80%-contained text-duplicate siblings."""
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

    # PDF font-render duplicates (#15): two textlines with identical
    # content placed at nearly the same coordinates — discard the
    # shorter (smaller-bbox) one as a true duplicate.
    big = _Box(0, 0, 100, 5, "SAME LINE")
    dup = _Box(10, 1, 80, 4, "SAME LINE")  # equal content, >80% inside big
    out = text_in_bbox((0, 0, 200, 10), [big, dup])
    assert big in out
    assert dup not in out

    # A single-character textline must NOT be classified as a duplicate
    # of a wider sibling that merely contains the letter (regression for
    # split_text=True lattice cells where 'B', 'C', etc. land next to
    # wider headers/labels).
    single = _Box(45, 1, 50, 4, "B")  # >80% inside a wider sibling below
    label = _Box(0, 0, 100, 5, "Category B")
    out_single = text_in_bbox((0, 0, 200, 10), [label, single])
    assert label in out_single
    assert single in out_single

    # Adjacent-cell text (#288, #625): a wide name overlaps a short
    # number that has nothing to do with it. The numeric textline must
    # survive — the old geometry-only rule dropped it incorrectly.
    name = _Box(0, 0, 100, 5, "Ackermann XXXXXXXXXXXXXXXXXX GmbH")
    number = _Box(70, 1, 95, 4, "11 111111111")  # >80% inside name's bbox
    out2 = text_in_bbox((0, 0, 200, 10), [name, number])
    assert name in out2
    assert number in out2

    # Empty input — empty output, no crash.
    assert text_in_bbox((0, 0, 100, 100), []) == []


# --- Coverage for #733: get_table_index NumPy / bisect refactor -------------


class _TextlineStub:
    """Minimal stand-in for a PDFMiner LTTextLine sufficient for get_table_index."""

    __slots__ = ("x0", "y0", "x1", "y1", "_objs", "_text")

    def __init__(self, x0, y0, x1, y1, text="x"):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self._objs = []
        self._text = text

    def get_text(self):
        return self._text + "\n"


def _make_textline(x0, y0, x1, y1, text="x"):
    """Build a _TextlineStub — kept as a thin helper for call-site readability."""
    return _TextlineStub(x0, y0, x1, y1, text)


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


def test_bbox_from_str_normalises_corner_order():
    # top-left/bottom-right and the inverted-y form both normalise to
    # (xmin, ymin, xmax, ymax) — no error for a valid, non-degenerate box.
    assert bbox_from_str("49,403,568,217") == (49, 217, 568, 403)
    assert bbox_from_str("49,217,568,403") == (49, 217, 568, 403)


def test_bbox_from_str_rejects_zero_area():
    # Zero width or height is the usual cause of the cryptic downstream
    # ZeroDivisionError (#63); it must raise a clear, hint-bearing error.
    for bad in ("49,217,568,217", "100,100,100,400"):
        with pytest.raises(ValueError, match="zero width or height"):
            bbox_from_str(bad)


def test_bbox_from_str_rejects_malformed():
    with pytest.raises(ValueError, match="four"):
        bbox_from_str("1,2,3")
    with pytest.raises(ValueError, match="must be numbers"):
        bbox_from_str("a,b,c,d")
