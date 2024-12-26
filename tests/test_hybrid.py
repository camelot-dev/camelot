import os

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot

from .data import *


def test_hybrid(testdir):
    df = pd.DataFrame(data_hybrid)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_table_rotated(testdir):
    df = pd.DataFrame(data_hybrid_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_two_tables(testdir):
    df1 = pd.DataFrame(data_network_two_tables_1)
    df2 = pd.DataFrame(data_network_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_hybrid_vertical_header(testdir):
    """Tests a complex table with a vertically text header."""
    df = pd.DataFrame(data_hybrid_vertical_headers)

    filename = os.path.join(testdir, "vertical_header.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", backend="pdfium", use_fallback=False
    )
    assert len(tables) == 1
    assert_frame_equal(df, tables[0].df)


def test_hybrid_process_background(testdir):
    df = pd.DataFrame(data_hybrid_process_background)

    filename = os.path.join(testdir, "background_lines_1.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", process_background=True)
    assert_frame_equal(df, tables[1].df)


def test_hybrid_table_regions(testdir):
    df = pd.DataFrame(data_network_table_regions)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="hybrid", table_regions=["320,335,573,505"]
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
    df = pd.DataFrame(data_network_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="hybrid",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_hybrid_flag_size(testdir):
    df = pd.DataFrame(data_network_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_hybrid_strip_text(testdir):
    df = pd.DataFrame(data_network_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_hybrid_edge_tol(testdir):
    df = pd.DataFrame(data_network_edge_tol)

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


def test_hybrid_keyerror(testdir):
    """Parsing this PDF generates a key error."""
    filename = os.path.join(testdir, "tabula/schools.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", pages="4-5")
    assert len(tables) >= 1
