# -*- coding: utf-8 -*-

import os
import sys

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.core import Table, TableList
from camelot.__version__ import generate_version

from .data import *

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")

skip_on_windows = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Ghostscript not installed in Windows test environment",
)


@skip_on_windows
def test_lattice():
    df = pd.DataFrame(data_lattice)

    filename = os.path.join(
        testdir, "tabula/icdar2013-dataset/competition-dataset-us/us-030.pdf"
    )
    tables = camelot.read_pdf(filename, pages="2")
    assert_frame_equal(df, tables[0].df)


@skip_on_windows
def test_lattice_table_rotated():
    df = pd.DataFrame(data_lattice_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)


@skip_on_windows
def test_lattice_two_tables():
    df1 = pd.DataFrame(data_lattice_two_tables_1)
    df2 = pd.DataFrame(data_lattice_two_tables_2)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename)
    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


@skip_on_windows
def test_lattice_table_regions():
    df = pd.DataFrame(data_lattice_table_regions)

    filename = os.path.join(testdir, "table_region.pdf")
    tables = camelot.read_pdf(filename, table_regions=["170,370,560,270"])
    assert_frame_equal(df, tables[0].df)


@skip_on_windows
def test_lattice_table_areas():
    df = pd.DataFrame(data_lattice_table_areas)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename, table_areas=["80,693,535,448"])
    assert_frame_equal(df, tables[0].df)


@skip_on_windows
def test_lattice_process_background():
    df = pd.DataFrame(data_lattice_process_background)

    filename = os.path.join(testdir, "background_lines_1.pdf")
    tables = camelot.read_pdf(filename, process_background=True)
    assert_frame_equal(df, tables[1].df)


@skip_on_windows
def test_lattice_copy_text():
    df = pd.DataFrame(data_lattice_copy_text)

    filename = os.path.join(testdir, "row_span_1.pdf")
    tables = camelot.read_pdf(filename, line_scale=60, copy_text="v")
    assert_frame_equal(df, tables[0].df)


@skip_on_windows
def test_lattice_shift_text():
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


@skip_on_windows
def test_lattice_arabic():
    df = pd.DataFrame(data_arabic)

    filename = os.path.join(testdir, "tabula/arabic.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)
