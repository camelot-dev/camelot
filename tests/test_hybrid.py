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


def test_hybrid_multipage(testdir):
    """Hybrid parser should clear table bboxes on each new page."""
    filename = os.path.join(testdir, "hybrid_multipage.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", pages="1-2")
    assert len(tables) == 2  # not 3


_ICDAR = "tabula/icdar2013-dataset"


def test_hybrid_gates_complete_ruled_grid_to_lattice(testdir):
    """A complete ruled grid is parsed by lattice, not over-split by network.

    Regression for the #38 over-split: hybrid used to *union* network's
    text-derived column splits onto an already-complete lattice grid and
    then parse the merged bbox with the network parser (text-grouped rows),
    wrecking fully-ruled tables (row accuracy collapsed). The completeness
    gate now routes a complete grid to the lattice parser untouched.
    """
    filename = os.path.join(testdir, _ICDAR, "competition-dataset-eu/eu-009a.pdf")
    hybrid = camelot.read_pdf(filename, flavor="hybrid")
    lattice = camelot.read_pdf(filename, flavor="lattice")
    network = camelot.read_pdf(filename, flavor="network")

    ruled_shape = lattice[0].df.shape  # (9, 4): the clean ruled grid
    hybrid_shapes = {t.df.shape for t in hybrid}
    # hybrid reproduces lattice's clean grid ...
    assert ruled_shape in hybrid_shapes
    # ... and that grid is not something the network parser produced on its
    # own, so the gate genuinely re-routed this table.
    assert ruled_shape not in {t.df.shape for t in network}


def test_hybrid_keeps_partial_ruled_grid_on_network(testdir):
    """A partially-ruled table stays on the network-augmented path.

    Here lattice only finds a small ruled fragment (the top rows); routing
    the whole table to it would silently drop the unruled rows. The
    completeness gate rejects such fragments, so hybrid keeps network's
    full-table extent — and must not regress this niche win.
    """
    filename = os.path.join(testdir, _ICDAR, "competition-dataset-us/us-008.pdf")
    hybrid = camelot.read_pdf(filename, flavor="hybrid")
    lattice = camelot.read_pdf(filename, flavor="lattice")
    network = camelot.read_pdf(filename, flavor="network")

    # hybrid keeps network's full table, not lattice's smaller fragment.
    assert hybrid[0].df.shape == network[0].df.shape
    assert hybrid[0].df.shape != lattice[0].df.shape


def test_hybrid_vector_engine_drops_empty_tables(testdir):
    """The render-free vector engine must not leak empty tables (#39).

    Vector ruled lines include decorative page borders / form rules that can
    raise a grid with no text inside it. eu-016's first page has such rules
    but no real table there; the empty grid must be rejected, not emitted.
    """
    filename = os.path.join(testdir, _ICDAR, "competition-dataset-eu/eu-016.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid", engine="vector")
    # no empty / degenerate tables leak out
    assert all(t.df.shape[0] > 0 and t.df.shape[1] > 0 for t in tables)
