# -*- coding: utf-8 -*-

import os

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.core import Table, TableList
from camelot.__version__ import generate_version

from .data import *

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


def test_stream():
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_table_rotated():
    df = pd.DataFrame(data_stream_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_two_tables():
    df1 = pd.DataFrame(data_stream_two_tables_1)
    df2 = pd.DataFrame(data_stream_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_stream_table_regions():
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_regions=["320,460,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_table_areas():
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_columns():
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", columns=["67,180,230,425,475"], row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_split_text():
    df = pd.DataFrame(data_stream_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_flag_size():
    df = pd.DataFrame(data_stream_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_stream_strip_text():
    df = pd.DataFrame(data_stream_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_stream_edge_tol():
    df = pd.DataFrame(data_stream_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", edge_tol=500)
    assert_frame_equal(df, tables[0].df)


def test_stream_layout_kwargs():
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_duplicated_text():
    df = pd.DataFrame(data_stream_duplicated_text)

    filename = os.path.join(testdir, "birdisland.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)
