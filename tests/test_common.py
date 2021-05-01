# -*- coding: utf-8 -*-

import os

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.core import Table, TableList
from camelot.__version__ import generate_version

from .data import *

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")

def test_parsing_report(parallel):
    parsing_report = {"accuracy": 99.02, "whitespace": 12.24, "order": 1, "page": 1}

    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert tables[0].parsing_report == parsing_report



def test_password(parallel):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health_protected.pdf")
    tables = camelot.read_pdf(filename, password="ownerpass", flavor="stream", parallel=parallel)
    assert_frame_equal(df, tables[0].df)

    tables = camelot.read_pdf(filename, password="userpass", flavor="stream", parallel=parallel)
    assert_frame_equal(df, tables[0].df)



def test_stream(parallel):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", parallel=parallel)
    assert_frame_equal(df, tables[0].df)



def test_stream_table_rotated(parallel):
    df = pd.DataFrame(data_stream_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", parallel=parallel)
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", parallel=parallel)
    assert_frame_equal(df, tables[0].df)



def test_stream_two_tables(parallel):
    df1 = pd.DataFrame(data_stream_two_tables_1)
    df2 = pd.DataFrame(data_stream_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", parallel=parallel)

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)



def test_stream_table_regions(parallel):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream",
        parallel=parallel,
        table_regions=["320,460,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_table_areas(parallel):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream",
        parallel=parallel,
        table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_columns(parallel):
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        parallel=parallel,
        columns=["67,180,230,425,475"],
        row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_split_text(parallel):
    df = pd.DataFrame(data_stream_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        parallel=parallel,
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_flag_size(parallel):
    df = pd.DataFrame(data_stream_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        parallel=parallel,
        flag_size=True
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_strip_text(parallel):
    df = pd.DataFrame(data_stream_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream",
        parallel=parallel,
        strip_text=" ,\n"
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_edge_tol(parallel):
    df = pd.DataFrame(data_stream_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream",
        parallel=parallel,
        edge_tol=500
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_layout_kwargs(parallel):
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        parallel=parallel,
        layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_lattice(parallel):
    df = pd.DataFrame(data_lattice)

    filename = os.path.join(
        testdir, "tabula/icdar2013-dataset/competition-dataset-us/us-030.pdf"
    )
    tables = camelot.read_pdf(filename, pages="2", parallel=parallel)
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_rotated(parallel):
    df = pd.DataFrame(data_lattice_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_1.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_1.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert_frame_equal(df, tables[0].df)


def test_lattice_two_tables(parallel):
    df1 = pd.DataFrame(data_lattice_two_tables_1)
    df2 = pd.DataFrame(data_lattice_two_tables_2)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_lattice_table_regions(parallel):
    df = pd.DataFrame(data_lattice_table_regions)

    filename = os.path.join(testdir, "table_region.pdf")
    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        table_regions=["170,370,560,270"]
    )
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_areas(parallel):
    df = pd.DataFrame(data_lattice_table_areas)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        table_areas=["80,693,535,448"]
    )
    assert_frame_equal(df, tables[0].df)


def test_lattice_process_background(parallel):
    df = pd.DataFrame(data_lattice_process_background)

    filename = os.path.join(testdir, "background_lines_1.pdf")
    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        process_background=True,
    )
    assert_frame_equal(df, tables[1].df)


def test_lattice_copy_text(parallel):
    df = pd.DataFrame(data_lattice_copy_text)

    filename = os.path.join(testdir, "row_span_1.pdf")
    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        line_scale=60,
        copy_text="v"
    )
    assert_frame_equal(df, tables[0].df)


def test_lattice_shift_text(parallel):
    df_lt = pd.DataFrame(data_lattice_shift_text_left_top)
    df_disable = pd.DataFrame(data_lattice_shift_text_disable)
    df_rb = pd.DataFrame(data_lattice_shift_text_right_bottom)

    filename = os.path.join(testdir, "column_span_2.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel, line_scale=40)
    assert df_lt.equals(tables[0].df)

    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        line_scale=40,
        shift_text=[""]
    )
    assert df_disable.equals(tables[0].df)

    tables = camelot.read_pdf(
        filename,
        parallel=parallel,
        line_scale=40,
        shift_text=["r", "b"]
    )
    assert df_rb.equals(tables[0].df)


def test_repr(parallel):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert (
        repr(tables[0].cells[0][0]) == "<Cell x1=120.48 y1=218.43 x2=164.64 y2=233.77>"
    )


def test_pages(parallel):
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, parallel=parallel)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert (
        repr(tables[0].cells[0][0]) == "<Cell x1=120.48 y1=218.43 x2=164.64 y2=233.77>"
    )

    tables = camelot.read_pdf(url, pages="1-end", parallel=parallel)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert (
        repr(tables[0].cells[0][0]) == "<Cell x1=120.48 y1=218.43 x2=164.64 y2=233.77>"
    )

    tables = camelot.read_pdf(url, pages="all", parallel=parallel)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert (
        repr(tables[0].cells[0][0]) == "<Cell x1=120.48 y1=218.43 x2=164.64 y2=233.77>"
    )


def test_url(parallel):
    url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
    tables = camelot.read_pdf(url, parallel=parallel)
    assert repr(tables) == "<TableList n=1>"
    assert repr(tables[0]) == "<Table shape=(7, 7)>"
    assert (
        repr(tables[0].cells[0][0]) == "<Cell x1=120.48 y1=218.43 x2=164.64 y2=233.77>"
    )


def test_arabic(parallel):
    df = pd.DataFrame(data_arabic)

    filename = os.path.join(testdir, "tabula/arabic.pdf")
    tables = camelot.read_pdf(filename, parallel=parallel)
    assert_frame_equal(df, tables[0].df)


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


def test_stream_duplicated_text():
    df = pd.DataFrame(data_stream_duplicated_text)

    filename = os.path.join(testdir, "birdisland.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)
