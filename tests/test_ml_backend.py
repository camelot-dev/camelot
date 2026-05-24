"""Torch-free tests for the ML (Table Transformer) backend — flavor='ml'.

These cover the pieces that do not need the optional ML dependencies:
the pure box->grid post-processing, the image->PDF coordinate mapping, the
spanning-cell edge application, parser registration, and the friendly error
raised when torch/transformers are missing. Live model inference is validated
separately once the [ml] extra is installed.
"""

import importlib.util

import pytest

from camelot.core import Table
from camelot.parsers.ml import DetectedObject
from camelot.parsers.ml import MachineLearning
from camelot.parsers.ml import apply_spans
from camelot.parsers.ml import grid_to_pdf_cols_rows
from camelot.parsers.ml import objects_to_grid

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _col(x0, x1, y0=0.0, y1=100.0, score=0.99):
    return DetectedObject("table column", score, (x0, y0, x1, y1))


def _row(y0, y1, x0=0.0, x1=100.0, score=0.99):
    return DetectedObject("table row", score, (x0, y0, x1, y1))


def _span(x0, y0, x1, y1, score=0.99):
    return DetectedObject("table spanning cell", score, (x0, y0, x1, y1))


# --------------------------------------------------------------------------- #
# objects_to_grid
# --------------------------------------------------------------------------- #
def test_objects_to_grid_clean_2x2():
    objects = [_col(0, 50), _col(50, 100), _row(0, 50), _row(50, 100)]
    grid = objects_to_grid(objects)
    assert grid.col_bounds == [(0.0, 50.0), (50.0, 100.0)]
    assert grid.row_bounds == [(0.0, 50.0), (50.0, 100.0)]
    assert grid.spans == []


def test_objects_to_grid_gap_split_at_midpoint():
    # Columns separated by a gap -> the separator lands in the gap centre.
    objects = [_col(0, 40), _col(60, 100), _row(0, 50), _row(50, 100)]
    grid = objects_to_grid(objects)
    assert grid.col_bounds == [(0.0, 50.0), (50.0, 100.0)]


def test_objects_to_grid_spanning_cell():
    objects = [
        _col(0, 30),
        _col(30, 60),
        _col(60, 90),
        _row(0, 40),
        _row(40, 80),
        _span(0, 0, 60, 40),  # row 0 spans columns 0 and 1
    ]
    grid = objects_to_grid(objects)
    assert len(grid.col_bounds) == 3
    assert len(grid.row_bounds) == 2
    assert grid.spans == [(0, 0, 0, 1)]


def test_objects_to_grid_score_threshold_drops_low_confidence():
    objects = [
        _col(0, 50),
        _col(50, 100, score=0.10),  # below threshold -> ignored
        _row(0, 100),
    ]
    grid = objects_to_grid(objects, score_thresh=0.5)
    assert len(grid.col_bounds) == 1


def test_objects_to_grid_merges_duplicate_bands():
    # Two near-identical detections of the same column collapse to one.
    objects = [_col(0, 50), _col(2, 52), _col(50, 100), _row(0, 100)]
    grid = objects_to_grid(objects, merge_tol=6.0)
    assert len(grid.col_bounds) == 2


def test_objects_to_grid_empty_without_rows_or_cols():
    assert objects_to_grid([]).col_bounds == []
    assert objects_to_grid([_col(0, 50)]).row_bounds == []  # cols but no rows
    assert objects_to_grid([_row(0, 50)]).col_bounds == []  # rows but no cols


# --------------------------------------------------------------------------- #
# grid_to_pdf_cols_rows
# --------------------------------------------------------------------------- #
def test_grid_to_pdf_cols_rows_flips_y_and_orders():
    grid = objects_to_grid([_col(0, 50), _col(50, 100), _row(0, 40), _row(40, 80)])
    # pdf_scalers = (sx, sy, image_height)
    cols, rows = grid_to_pdf_cols_rows(grid, (0.5, 0.5, 80))
    # x just scales; columns stay increasing.
    assert cols == [(0.0, 25.0), (25.0, 50.0)]
    # y flips (image top-left -> pdf bottom-left): rows decreasing by y_top.
    assert rows == [(40.0, 20.0), (20.0, 0.0)]
    assert rows[0][0] > rows[1][0]  # top row sits higher in PDF space


# --------------------------------------------------------------------------- #
# apply_spans
# --------------------------------------------------------------------------- #
def test_apply_spans_opens_interior_edges():
    table = Table([(0, 1), (1, 2), (2, 3)], [(1, 0)]).set_all_edges()
    apply_spans(table, [(0, 0, 0, 1)])  # merge the first two cells of row 0
    assert table.cells[0][0].hspan  # right edge opened
    assert table.cells[0][1].hspan  # left edge opened
    assert not table.cells[0][2].hspan  # untouched cell stays bounded


# --------------------------------------------------------------------------- #
# registration + wiring
# --------------------------------------------------------------------------- #
def test_ml_flavor_registered():
    from camelot.handlers import PARSERS
    from camelot.utils import flavor_to_kwargs
    from camelot.utils import validate_input

    assert PARSERS["ml"] is MachineLearning
    assert "ml" in flavor_to_kwargs
    # Common + ML-specific kwargs validate; an unrelated one is rejected.
    validate_input({"split_text": True, "device": "cpu"}, flavor="ml")
    with pytest.raises(ValueError, match="cannot be used with flavor='ml'"):
        validate_input({"line_scale": 40}, flavor="ml")


def test_ml_constructs_without_torch():
    parser = MachineLearning(device="cpu", structure_threshold=0.7)
    assert parser.id == "machine_learning"
    assert parser.structure_threshold == 0.7
    assert parser.shift_text == ["l", "t"]


@pytest.mark.skipif(_HAS_TORCH, reason="torch installed; the import guard is moot")
def test_ml_without_deps_raises_helpful_error():
    parser = MachineLearning()
    with pytest.raises(ImportError, match=r"camelot-py\[ml\]"):
        parser._load_models()


@pytest.mark.skipif(_HAS_TORCH, reason="torch installed; would fetch real models")
def test_ml_read_pdf_surfaces_install_hint(foo_pdf):
    """End-to-end dispatch wiring: read_pdf -> handler -> ML parser -> hint."""
    import camelot

    with pytest.raises(ImportError, match=r"camelot-py\[ml\]"):
        camelot.read_pdf(foo_pdf, flavor="ml")
