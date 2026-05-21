"""TableList.stack_contiguous: vertically stack tables across pages (#628).

Consolidates #8 / #133 / #357 / #531 / #628 — vertically concatenate
tables that look like continuations across page breaks.
"""

import pandas as pd
import pytest

from camelot.core import Table
from camelot.core import TableList


def _make_table(df, page=1, order=1, cols_x=(0, 100, 200), rows_y=(300, 200, 100)):
    """Build a minimal Table whose .df and column count match.

    The exact y-coords of rows/cells don't matter for stacking semantics;
    we just need them to round-trip through the shift logic.
    """
    cols = list(zip(cols_x[:-1], cols_x[1:], strict=False))
    rows = list(zip(rows_y[:-1], rows_y[1:], strict=False))
    t = Table(cols=cols, rows=rows)
    t.df = df.reset_index(drop=True)
    t.page = page
    t.order = order
    t.accuracy = 95.0
    t.whitespace = 5.0
    # Real parser-built Tables carry a _bbox; synthetic ones default to
    # None, which the geometry-shift in _vstack_run would index into.
    # Set it from the cols/rows extent so the stacking math runs.
    t._bbox = (min(cols_x), min(rows_y), max(cols_x), max(rows_y))
    # `confidence` is a read-only @property on Table; it recomputes
    # from accuracy + whitespace on access.
    return t


def test_empty_tablelist_stack_returns_empty():
    out = TableList([]).stack_contiguous()
    assert len(out) == 0


def test_single_table_stack_is_passthrough_copy():
    df = pd.DataFrame([["a", "b"], ["c", "d"]])
    t = _make_table(df)
    stacked = TableList([t]).stack_contiguous()
    assert len(stacked) == 1
    # Deep-copy: mutating the result doesn't touch the source.
    stacked[0].df.iloc[0, 0] = "X"
    assert t.df.iloc[0, 0] == "a"


def test_two_tables_same_columns_get_stacked():
    df1 = pd.DataFrame([["1", "Alice"], ["2", "Bob"]])
    df2 = pd.DataFrame([["3", "Charlie"], ["4", "Diana"]])
    stacked = TableList(
        [_make_table(df1, page=1), _make_table(df2, page=2)]
    ).stack_contiguous()
    assert len(stacked) == 1
    result = stacked[0].df
    assert result.shape == (4, 2)
    assert list(result.iloc[:, 1]) == ["Alice", "Bob", "Charlie", "Diana"]


def test_two_tables_different_columns_not_stacked():
    df1 = pd.DataFrame([["a", "b"]])
    df2 = pd.DataFrame([["c", "d", "e"]])  # different ncols
    stacked = TableList(
        [_make_table(df1), _make_table(df2, cols_x=(0, 100, 200, 300))]
    ).stack_contiguous()
    assert len(stacked) == 2


def test_three_tables_partial_stacking():
    # cols=2, cols=2, cols=3 → stack first two, third stays separate.
    df1 = pd.DataFrame([["1", "Alice"]])
    df2 = pd.DataFrame([["2", "Bob"]])
    df3 = pd.DataFrame([["a", "b", "c"]])
    stacked = TableList(
        [
            _make_table(df1),
            _make_table(df2),
            _make_table(df3, cols_x=(0, 100, 200, 300)),
        ]
    ).stack_contiguous()
    assert len(stacked) == 2
    assert stacked[0].df.shape == (2, 2)
    assert stacked[1].df.shape == (1, 3)


def test_match_first_row_keeps_only_first_header():
    header = ["ID", "Name"]
    df1 = pd.DataFrame([header, ["1", "Alice"], ["2", "Bob"]])
    df2 = pd.DataFrame([header, ["3", "Charlie"]])
    stacked = TableList([_make_table(df1), _make_table(df2)]).stack_contiguous(
        match="first_row"
    )
    assert len(stacked) == 1
    # Continuation table's header row should have been dropped, leaving one header.
    result = stacked[0].df
    assert list(result.iloc[0]) == header
    assert result.shape == (4, 2)
    # No second 'ID'/'Name' header lurking in the body.
    assert "ID" not in result.iloc[1:].values.flatten().tolist()


def test_match_first_row_keep_repeated_header_when_requested():
    header = ["ID", "Name"]
    df1 = pd.DataFrame([header, ["1", "Alice"]])
    df2 = pd.DataFrame([header, ["2", "Bob"]])
    stacked = TableList([_make_table(df1), _make_table(df2)]).stack_contiguous(
        match="first_row", keep_first_header=True
    )
    assert stacked[0].df.shape == (4, 2)


def test_match_first_row_distinct_headers_not_stacked():
    df1 = pd.DataFrame([["ID", "Name"], ["1", "Alice"]])
    df2 = pd.DataFrame([["NUM", "WHO"], ["2", "Bob"]])  # different header text
    stacked = TableList([_make_table(df1), _make_table(df2)]).stack_contiguous(
        match="first_row"
    )
    assert len(stacked) == 2  # Different headers → not continuations.


def test_stack_invalid_match_raises():
    with pytest.raises(ValueError, match="must be 'column_count' or 'first_row'"):
        TableList([]).stack_contiguous(match="bogus")


def test_stack_preserves_first_tables_page_and_order():
    df1 = pd.DataFrame([["1", "Alice"]])
    df2 = pd.DataFrame([["2", "Bob"]])
    stacked = TableList(
        [
            _make_table(df1, page=3, order=2),
            _make_table(df2, page=4, order=1),
        ]
    ).stack_contiguous()
    assert stacked[0].page == 3
    assert stacked[0].order == 2


def test_stack_averages_quality_metrics():
    df = pd.DataFrame([["x"]])
    t1 = _make_table(df, cols_x=(0, 100))
    t2 = _make_table(df, cols_x=(0, 100))
    t1.accuracy, t1.whitespace = 80.0, 10.0
    t2.accuracy, t2.whitespace = 100.0, 0.0
    stacked = TableList([t1, t2]).stack_contiguous()
    assert stacked[0].accuracy == pytest.approx(90.0)
    assert stacked[0].whitespace == pytest.approx(5.0)
    # confidence = (acc/100) * (1 - ws/100) = 0.9 * 0.95 = 0.855
    assert stacked[0].confidence == pytest.approx(0.855)


def test_stack_does_not_mutate_input_tables():
    df1 = pd.DataFrame([["1", "Alice"]])
    df2 = pd.DataFrame([["2", "Bob"]])
    t1 = _make_table(df1)
    t2 = _make_table(df2)
    original_t2_rows = list(t2.rows)
    _ = TableList([t1, t2]).stack_contiguous()
    assert t2.rows == original_t2_rows  # input unchanged
    assert t1.df.shape == (1, 2)
