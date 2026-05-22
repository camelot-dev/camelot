import os

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot

from .data import *


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


def test_lattice_arabic(testdir):
    df = pd.DataFrame(data_arabic)

    filename = os.path.join(testdir, "tabula/arabic.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)


def test_lattice_split_text(testdir):
    df = pd.DataFrame(data_lattice_split_text)

    filename = os.path.join(testdir, "split_text_lattice.pdf")
    tables = camelot.read_pdf(filename, line_scale=60, split_text=True)

    assert_frame_equal(df, tables[0].df)


def test_lattice_rejects_mostly_empty_grid():
    # #36 precision gate: a near-empty ruled grid is detection noise, not a
    # table, and must be dropped; a normally-filled table is kept.
    from types import SimpleNamespace

    from camelot.parsers.lattice import _GRID_WHITESPACE_REJECT
    from camelot.parsers.lattice import Lattice

    parser = Lattice()
    assert parser._reject_table(SimpleNamespace(whitespace=_GRID_WHITESPACE_REJECT + 1))
    assert not parser._reject_table(SimpleNamespace(whitespace=40.0))


def test_network_keeps_sparse_tables():
    # The gate is lattice-only — text-based parsers must not inherit it.
    from types import SimpleNamespace

    from camelot.parsers import Network

    assert not Network()._reject_table(SimpleNamespace(whitespace=99.0))


def test_lattice_engine_default_is_combined():
    # #763 / item-2: 'combined' is the default lattice engine.
    from camelot.parsers import Lattice

    assert Lattice().engine == "combined"


def test_lattice_engine_auto_rejected():
    # engine='auto' was dropped in the flavor x engine cleanup.
    import pytest

    from camelot.parsers import Lattice

    with pytest.raises(ValueError, match="engine must be"):
        Lattice(engine="auto")


def test_engine_rejected_for_non_lattice_flavor(testdir):
    # engine is lattice-only; passing it to a text-based flavor errors.
    import pytest

    import camelot

    filename = os.path.join(testdir, "foo.pdf")
    with pytest.raises(ValueError, match="engine"):
        camelot.read_pdf(filename, flavor="network", engine="combined")


def test_combined_respects_table_regions(testdir):
    # Regression: combined's vector lines must be clipped to table_regions,
    # so it never expands the table beyond the region vs raster.
    import camelot

    filename = os.path.join(testdir, "table_region.pdf")
    region = ["170,370,560,270"]
    r = camelot.read_pdf(
        filename, table_regions=region, engine="raster", suppress_stdout=True
    )
    c = camelot.read_pdf(
        filename, table_regions=region, engine="combined", suppress_stdout=True
    )
    assert c[0].shape == r[0].shape
