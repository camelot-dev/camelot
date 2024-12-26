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
    parsing_report = {"accuracy": 99.02, "whitespace": 12.24, "order": 1, "page": 1}

    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    assert tables[0].parsing_report == parsing_report


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
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
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
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
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


def test_handler_with_stream(testdir):
    filename = os.path.join(testdir, "foo.pdf")

    with open(filename, "rb") as f:
        handler = PDFHandler(f)
        assert handler._get_pages("1") == [1]


def test_handler_with_pathlib(testdir):
    filename = Path(os.path.join(testdir, "foo.pdf"))

    with open(filename, "rb") as f:
        handler = PDFHandler(f)
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
