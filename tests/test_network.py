import os

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot

from .data import *


# this one does not increase coverage
def test_network(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    assert_frame_equal(df, tables[0].df)


def test_network_table_rotated(testdir):
    df = pd.DataFrame(data_network_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    assert_frame_equal(df, tables[0].df)


def test_network_two_tables_a(testdir):
    df1 = pd.DataFrame(data_network_two_tables_1)
    df2 = pd.DataFrame(data_network_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    # tables = camelot.read_pdf(filename, flavor="hybrid")  # temp try hybrid

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


# Reported as https://github.com/camelot-dev/camelot/issues/132
def test_network_two_tables_b(testdir):
    df1 = pd.DataFrame(data_network_two_tables_b_1)
    df2 = pd.DataFrame(data_network_two_tables_b_2)
    filename = os.path.join(testdir, "multiple_tables.pdf")
    tables = camelot.read_pdf(filename, flavor="network")  # temp try hybrid

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_network_vertical_header(testdir):
    """Tests a complex table with a vertically text header."""
    df = pd.DataFrame(data_network_vertical_headers)
    filename = os.path.join(testdir, "vertical_header.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    assert len(tables) == 1
    assert_frame_equal(df, tables[0].df)


def test_network_table_regions(testdir):
    df = pd.DataFrame(data_network_table_regions)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    # The "stream" test looks for a region in ["320,460,573,335"], which
    # should exclude the header.
    tables = camelot.read_pdf(
        filename, flavor="network", table_regions=["320,335,573,505"]
    )
    assert_frame_equal(df, tables[0].df)


def test_network_table_areas(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="network", table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_network_columns(testdir):
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename, flavor="network", columns=["67,180,230,425,475"], row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_network_split_text(testdir):
    df = pd.DataFrame(data_network_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="network",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_network_flag_size(testdir):
    df = pd.DataFrame(data_network_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="network", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_network_strip_text(testdir):
    df = pd.DataFrame(data_network_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="network", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_network_edge_tol(testdir):
    df = pd.DataFrame(data_network_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(filename, flavor="network", edge_tol=500)
    assert_frame_equal(df, tables[0].df)


def test_network_layout_kwargs(testdir):
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="network", layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_network_no_infinite_execution(testdir):
    """Test for not infinite execution.

    This test used to fail, because the network parse was'nt able to process the tables on this pages.
    After a refactor it stops infinite execution. But parsing result could be improved.
    Hence this is no qualitative test.
    """
    filename = os.path.join(testdir, "tabula/schools.pdf")
    tables = camelot.read_pdf(
        filename, flavor="network", backend="ghostscript", pages="4"
    )

    assert len(tables) >= 1
