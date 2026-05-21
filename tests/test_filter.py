from camelot.core import Table
from camelot.core import TableList


def _table(rows, cols, accuracy=99.0, whitespace=0.0):
    # Minimal Table: filter() only reads .shape / .accuracy / .whitespace,
    # so set those directly (real cell geometry is irrelevant here).
    t = Table(cols=[(0, 1)], rows=[(1, 0)])
    t.shape = (rows, cols)
    t.accuracy = accuracy
    t.whitespace = whitespace
    return t


def test_filter_defaults_keep_everything():
    tables = TableList([_table(1, 1), _table(5, 3)])
    out = tables.filter()
    assert isinstance(out, TableList)
    assert out.n == 2


def test_filter_min_rows_and_columns():
    tables = TableList([_table(1, 4), _table(4, 1), _table(4, 4)])
    assert tables.filter(min_rows=2).n == 2  # drops the 1-row table
    assert tables.filter(min_columns=2).n == 2  # drops the 1-col table
    assert tables.filter(min_rows=2, min_columns=2).n == 1  # only 4x4


def test_filter_accuracy_and_whitespace():
    tables = TableList(
        [
            _table(3, 3, accuracy=95.0, whitespace=10.0),
            _table(3, 3, accuracy=40.0, whitespace=10.0),
            _table(3, 3, accuracy=95.0, whitespace=80.0),
        ]
    )
    assert tables.filter(min_accuracy=80.0).n == 2
    assert tables.filter(max_whitespace=50.0).n == 2
    assert tables.filter(min_accuracy=80.0, max_whitespace=50.0).n == 1


def test_filter_returns_new_list_and_composes():
    tables = TableList([_table(1, 1), _table(5, 5, accuracy=99.0)])
    filtered = tables.filter(min_rows=2).filter(min_accuracy=90.0)
    assert filtered.n == 1
    # original untouched
    assert tables.n == 2


def test_filter_thresholds_are_inclusive():
    tables = TableList([_table(2, 2, accuracy=80.0, whitespace=50.0)])
    # boundary values must be kept (>= / <=)
    assert (
        tables.filter(
            min_rows=2, min_columns=2, min_accuracy=80.0, max_whitespace=50.0
        ).n
        == 1
    )
