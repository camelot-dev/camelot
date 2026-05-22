import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.backends.ghostscript_backend import GhostscriptBackend
from camelot.core import Table
from camelot.core import TableList
from camelot.io import PDFHandler

from .conftest import skip_on_windows
from .conftest import skip_pdftopng
from .data import *


@skip_on_windows
def test_parsing_report(testdir):
    # #659: parsing_report also now includes a 'confidence' composite in
    # [0, 1] computed from accuracy + whitespace. The other keys are
    # unchanged; this test keeps pinning those values and adds the
    # derived 'confidence' alongside.
    expected = {
        "accuracy": 99.02,
        "whitespace": 12.24,
        "order": 1,
        "page": 1,
        "confidence": round((99.02 / 100.0) * (1.0 - 12.24 / 100.0), 4),
    }

    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    assert tables[0].parsing_report == expected


def test_password(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health_protected.pdf")
    tables = camelot.read_pdf(filename, password="ownerpass", flavor="stream")
    assert_frame_equal(df, tables[0].df)

    tables = camelot.read_pdf(filename, password="userpass", flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_repr_pdfium(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(
        filename, flavor="lattice", backend="pdfium", use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=121 y1=218 x2=165 y2=234>"


@skip_pdftopng
def test_repr_poppler(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


@skip_on_windows
def test_repr_ghostscript(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


@skip_on_windows
def test_repr_ghostscript_custom_backend(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend=GhostscriptBackend())
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


def test_url_pdfium():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(
        url, flavor="lattice", backend="pdfium", use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=121 y1=218 x2=165 y2=234>"


@skip_pdftopng
def test_url_poppler():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


@skip_on_windows
def test_url_ghostscript(testdir):
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="ghostscript")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


@skip_on_windows
def test_url_ghostscript_custom_backend(testdir):
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend=GhostscriptBackend())
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


def test_pages_pdfium():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="pdfium", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=121 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="1-end", backend="pdfium", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=121 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="all", backend="pdfium", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=121 y1=218 x2=165 y2=234>"


@skip_pdftopng
def test_pages_poppler():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="poppler", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="1-end", backend="poppler", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="all", backend="poppler", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


@skip_on_windows
def test_pages_ghostscript():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="ghostscript", use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url, pages="1-end", backend="ghostscript", use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url, pages="all", backend="ghostscript", use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


@skip_on_windows
def test_pages_ghostscript_custom_backend():
    url = "https://camelot-py.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    custom_backend = GhostscriptBackend()
    tables = camelot.read_pdf(url, backend=custom_backend, use_fallback=False)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url, pages="1-end", backend=custom_backend, use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url, pages="all", backend=custom_backend, use_fallback=False
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


def test_table_order():
    def _make_table(page, order):
        t = Table([], [])
        t.page = page
        t.order = order
        return t

    table_list = TableList(
        [_make_table(2, 1), _make_table(1, 1), _make_table(3, 4), _make_table(1, 2)]
    )

    assert [(t.page, t.order) for t in sorted(table_list)] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 4),
    ]
    assert [(t.page, t.order) for t in sorted(table_list, reverse=True)] == [
        (3, 4),
        (2, 1),
        (1, 2),
        (1, 1),
    ]


def test_handler_pages_generator(testdir):
    filename = os.path.join(testdir, "foo.pdf")

    handler = PDFHandler(filename)
    assert handler._get_pages("1") == [1]

    handler = PDFHandler(filename)
    assert handler._get_pages("all") == [1]

    handler = PDFHandler(filename)
    assert handler._get_pages("1-end") == [1]

    handler = PDFHandler(filename)
    assert handler._get_pages("1,2,3,4") == [1, 2, 3, 4]

    handler = PDFHandler(filename)
    assert handler._get_pages("1,2,5-10") == [1, 2, 5, 6, 7, 8, 9, 10]

    handler = PDFHandler(
        os.path.join(testdir, "health_protected.pdf"), password="ownerpass"
    )

    assert handler._get_pages("all") == [1]


def test_handler_with_pathlib(testdir):
    filename = Path(os.path.join(testdir, "foo.pdf"))
    handler = PDFHandler(filename)
    assert handler._get_pages("1") == [1]


def test_table_list_iter():
    def _make_table(page, order):
        t = Table([], [])
        t.page = page
        t.order = order
        return t

    table_list = TableList(
        [_make_table(2, 1), _make_table(1, 1), _make_table(3, 4), _make_table(1, 2)]
    )
    # https://docs.python.org/3.12/library/functions.html#iter
    # https://docs.python.org/3.12/library/stdtypes.html#typeiter
    iterator_a = iter(table_list)
    assert iterator_a is not None
    item_a = next(iterator_a)
    assert item_a is not None

    item_b = table_list.__getitem__(0)
    assert item_b is not None

    iterator_b = table_list.__iter__()
    assert iterator_b is not None
    item_c = next(iterator_b)
    assert item_c is not None


def test_tablelist_accepts_iterable():
    """Accept any Iterable[Table] for TableList — not just Sized (#655)."""
    from unittest.mock import MagicMock

    # Empty generator: bool / len must work without consuming-issues.
    empty = TableList(t for t in [])
    assert len(empty) == 0
    assert bool(empty) is False

    # Non-empty generator: previously raised TypeError on len/bool. Use
    # MagicMock(spec=Table) so the typeguard session is happy with
    # __getitem__'s `-> Table` annotation; plain `object()` would trip
    # `typeguard.TypeCheckError: the return value (object) is not an
    # instance of camelot.core.Table`.
    sentinels = [MagicMock(spec=Table), MagicMock(spec=Table)]
    tl = TableList(iter(sentinels))
    assert len(tl) == 2
    assert bool(tl) is True
    assert tl[0] is sentinels[0]
    assert tl[1] is sentinels[1]


def test_parsing_report_includes_confidence():
    """parsing_report now includes a unified 'confidence' composite in [0, 1] (#659)."""
    import math

    from camelot.core import Table

    t = Table([(0.0, 100.0), (100.0, 200.0)], [(100.0, 90.0), (90.0, 80.0)])
    t.accuracy = 90.0
    t.whitespace = 10.0
    t.order = 1
    t.page = 1

    report = t.parsing_report
    # Schema: legacy keys still there, new key added.
    assert {"page", "order", "accuracy", "whitespace", "confidence"} == set(report)
    assert math.isclose(report["confidence"], 0.81, abs_tol=1e-4)
    assert 0.0 <= report["confidence"] <= 1.0

    # Edge cases.
    t.accuracy, t.whitespace = 100.0, 0.0
    assert math.isclose(t.confidence, 1.0)
    t.accuracy, t.whitespace = 0.0, 50.0
    assert t.confidence == 0.0
    t.accuracy, t.whitespace = 100.0, 100.0
    assert t.confidence == 0.0


def test_auto_flavor_warns_and_extracts(testdir):
    """flavor='auto' picks one of the supported flavors, warns, and parses."""
    import re
    import warnings

    import camelot

    filename = os.path.join(testdir, "foo.pdf")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tables = camelot.read_pdf(filename, flavor="auto")
    # The auto-detection warning must fire with one of the supported flavor
    # names. We deliberately don't pin which one — the heuristic depends on
    # the rendered-image line-detection sensitivity, which varies slightly
    # across cv2/Pillow builds. What matters is that auto returned a real
    # flavor (not 'auto' echoed back), the warning fired, and the parse
    # produced at least one table on foo.pdf.
    auto_warns = [w for w in caught if "auto-detected" in str(w.message)]
    assert auto_warns, f"expected an auto-detection UserWarning, got: {caught!r}"
    msg = str(auto_warns[-1].message)
    # auto now reports its choice(s) per page, e.g.
    # "auto-detected per-page flavors {1: 'lattice'}".
    flavors = set(re.findall(r"'(lattice|stream|network|hybrid)'", msg))
    assert flavors, f"unexpected warning text: {msg!r}"
    assert flavors <= {"lattice", "stream", "network", "hybrid"}
    assert len(tables) >= 1


def test_auto_flavor_rejects_unknown():
    """flavor='unknown' still raises NotImplementedError with the new value listed."""
    import pytest

    import camelot

    with pytest.raises(NotImplementedError, match=r"auto"):
        camelot.read_pdf("nonexistent.pdf", flavor="banana")
