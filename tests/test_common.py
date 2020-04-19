import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
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


def test_stream(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_table_rotated(testdir):
    df = pd.DataFrame(data_stream_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_two_tables(testdir):
    df1 = pd.DataFrame(data_stream_two_tables_1)
    df2 = pd.DataFrame(data_stream_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_stream_table_regions(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_regions=["320,460,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_table_areas(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_columns(testdir):
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", columns=["67,180,230,425,475"], row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_split_text(testdir):
    df = pd.DataFrame(data_stream_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_flag_size(testdir):
    df = pd.DataFrame(data_stream_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_stream_strip_text(testdir):
    df = pd.DataFrame(data_stream_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_stream_edge_tol(testdir):
    df = pd.DataFrame(data_stream_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", edge_tol=500)
    assert_frame_equal(df, tables[0].df)


def test_stream_layout_kwargs(testdir):
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_table_rotated(testdir):
    df = pd.DataFrame(data_stream_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_two_tables(testdir):
    df1 = pd.DataFrame(data_stream_two_tables_1)
    df2 = pd.DataFrame(data_stream_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_hybrid_table_regions(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", table_regions=["320,460,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid_table_areas(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid_columns(testdir):
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", columns=["67,180,230,425,475"], row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid_split_text(testdir):
    df = pd.DataFrame(data_stream_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="hybrid",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid_flag_size(testdir):
    df = pd.DataFrame(data_stream_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_hybrid_strip_text(testdir):
    df = pd.DataFrame(data_stream_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_edge_tol(testdir):
    df = pd.DataFrame(data_stream_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", edge_tol=500)
    assert_frame_equal(df, tables[0].df)


def test_hybrid_layout_kwargs(testdir):
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_lattice(testdir):
    df = pd.DataFrame(data_lattice)

    filename = os.path.join(
        testdir, "tabula/icdar2013-dataset/competition-dataset-us/us-030.pdf"
    )
    tables = camelot.read_pdf(filename, pages="2")
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_rotated(testdir):
    df = pd.DataFrame(data_lattice_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)


def test_lattice_two_tables(testdir):
    df1 = pd.DataFrame(data_lattice_two_tables_1)
    df2 = pd.DataFrame(data_lattice_two_tables_2)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename)
    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_lattice_table_regions(testdir):
    df = pd.DataFrame(data_lattice_table_regions)

    filename = os.path.join(testdir, "table_region.pdf")
    tables = camelot.read_pdf(filename, table_regions=["170,370,560,270"])
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_areas(testdir):
    df = pd.DataFrame(data_lattice_table_areas)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename, table_areas=["80,693,535,448"])
    assert_frame_equal(df, tables[0].df)


def test_lattice_process_background(testdir):
    df = pd.DataFrame(data_lattice_process_background)

    filename = os.path.join(testdir, "background_lines_1.pdf")
    tables = camelot.read_pdf(filename, process_background=True)
    assert_frame_equal(df, tables[1].df)


def test_lattice_copy_text(testdir):
    df = pd.DataFrame(data_lattice_copy_text)

    filename = os.path.join(testdir, "row_span_1.pdf")
    tables = camelot.read_pdf(filename, line_scale=60, copy_text="v")
    assert_frame_equal(df, tables[0].df)


def test_lattice_shift_text(testdir):
    df_lt = pd.DataFrame(data_lattice_shift_text_left_top)
    df_disable = pd.DataFrame(data_lattice_shift_text_disable)
    df_rb = pd.DataFrame(data_lattice_shift_text_right_bottom)

    filename = os.path.join(testdir, "column_span_2.pdf")
    tables = camelot.read_pdf(filename, line_scale=40)
    assert df_lt.equals(tables[0].df)

    tables = camelot.read_pdf(filename, line_scale=40, shift_text=[""])
    assert df_disable.equals(tables[0].df)

    tables = camelot.read_pdf(filename, line_scale=40, shift_text=["r", "b"])
    assert df_rb.equals(tables[0].df)


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


@skip_pdftopng
def test_url_poppler():
    url = "https://pypdf-table-extraction.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


@skip_on_windows
def test_url_ghostscript(testdir):
    url = "https://pypdf-table-extraction.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="ghostscript")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"


@skip_pdftopng
def test_pages_poppler():
    url = "https://pypdf-table-extraction.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="1-end", backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="all", backend="poppler")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=219 x2=165 y2=234>"


@skip_on_windows
def test_pages_ghostscript():
    url = "https://pypdf-table-extraction.readthedocs.io/en/latest/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, backend="ghostscript")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="1-end", backend="ghostscript")
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert repr(tables[0].cells[0][0]) == "<Cell x1=120 y1=218 x2=165 y2=234>"

    tables = camelot.read_pdf(url, pages="all", backend="ghostscript")
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
