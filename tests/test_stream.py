import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import camelot

from .data import *


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


def test_stream_header_footer_text(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        header_text=["Public Health Outlay"],
        footer_text=["Health Sector Financing"],
    )

    assert len(tables) == 1
    table = tables[0]
    assert table._bbox[0] == pytest.approx(0)
    assert table._bbox[1] > 0
    assert table._bbox[2] == pytest.approx(table.pdf_size[0])
    assert table._bbox[3] < table.pdf_size[1]
    assert_frame_equal(df, table.df)
    table_text = table.df.to_string()
    assert "Public Health Outlay" not in table_text
    assert "Health Sector Financing" not in table_text


def test_stream_header_text_only_defaults_to_page_bottom(testdir):
    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        header_text=["Public Health Outlay"],
    )

    assert len(tables) == 1
    table = tables[0]
    assert table._bbox[0] == pytest.approx(0)
    assert table._bbox[1] == pytest.approx(0)
    assert table._bbox[2] == pytest.approx(table.pdf_size[0])
    assert table._bbox[3] < table.pdf_size[1]


def test_stream_missing_header_text_falls_back_to_auto_detection(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        header_text=["not a real heading on this page"],
    )

    assert len(tables) == 1
    assert_frame_equal(df, tables[0].df)


def test_stream_table_areas_take_precedence_over_header_text(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        table_areas=["320,500,573,335"],
        header_text=["Instructions."],
    )

    assert_frame_equal(df, tables[0].df)


def test_header_text_rejected_for_lattice(testdir):
    filename = os.path.join(testdir, "health.pdf")
    with pytest.raises(ValueError, match="header_text cannot be used"):
        camelot.read_pdf(filename, header_text=["Public Health Outlay"])


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


def test_stream_inner_outer_columns(testdir):
    df = pd.DataFrame(data_stream_inner_outer_columns)

    filename = os.path.join(testdir, "stream_inner_outer_columns.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_fewer_columns_than_tables(testdir):
    """One column spec applied to a multi-table page no longer crashes (#112).

    Previously, supplying a `columns=` list shorter than the number of
    auto-detected tables raised IndexError on the second table. The last
    entry is now reused as a fallback so the call returns a result for
    every detected table.
    """
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        columns=["72,95,209,327,442,520"],
    )
    assert len(tables) == 2
