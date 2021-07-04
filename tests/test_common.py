# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.core import Table, TableList
from camelot.__version__ import generate_version
from camelot.backends import ImageConversionBackend

from .data import *

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


def test_version_generation():
    version = (0, 7, 3)
    assert generate_version(version, prerelease=None, revision=None) == "0.7.3"


def test_version_generation_with_prerelease_revision():
    version = (0, 7, 3)
    prerelease = "alpha"
    revision = 2
    assert (
        generate_version(version, prerelease=prerelease, revision=revision)
        == "0.7.3-alpha.2"
    )


def test_parsing_report():
    parsing_report = {"accuracy": 99.02, "whitespace": 12.24, "order": 1, "page": 1}

    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    assert tables[0].parsing_report == parsing_report


def test_password():
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health_protected.pdf")
    tables = camelot.read_pdf(filename, password="ownerpass", flavor="stream")
    assert_frame_equal(df, tables[0].df)

    tables = camelot.read_pdf(filename, password="userpass", flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_repr_poppler():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(
        filename, backend=ImageConversionBackend(backend="poppler", use_fallback=False)
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


def test_repr_ghostscript():
    if sys.platform not in ["linux", "darwin"]:
        return True

    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(
        filename,
        backend=ImageConversionBackend(backend="ghostscript", use_fallback=False),
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


def test_url_poppler():
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(
        url, backend=ImageConversionBackend(backend="poppler", use_fallback=False)
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


def test_url_ghostscript():
    if sys.platform not in ["linux", "darwin"]:
        return True

    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(
        url, backend=ImageConversionBackend(backend="ghostscript", use_fallback=False)
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


def test_pages_poppler():
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(
        url, backend=ImageConversionBackend(backend="poppler", use_fallback=False)
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url,
        pages="1-end",
        backend=ImageConversionBackend(backend="poppler", use_fallback=False),
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url,
        pages="all",
        backend=ImageConversionBackend(backend="poppler", use_fallback=False),
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


def test_pages_ghostscript():
    if sys.platform not in ["linux", "darwin"]:
        return True

    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(
        url, backend=ImageConversionBackend(backend="ghostscript", use_fallback=False)
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url,
        pages="1-end",
        backend=ImageConversionBackend(backend="ghostscript", use_fallback=False),
    )
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(
        url,
        pages="all",
        backend=ImageConversionBackend(backend="ghostscript", use_fallback=False),
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
