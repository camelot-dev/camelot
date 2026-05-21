import sqlite3

import pandas as pd
import pytest

from camelot.core import Table
from camelot.core import TableList


def _table(data, page=1, order=1):
    # Minimal parser-free Table: real cols/rows so Cell construction works,
    # plus the df / page / order the export paths read.
    cols = [(0, 100), (100, 200)]
    rows = [(200, 100), (100, 0)]
    t = Table(cols=cols, rows=rows)
    t.df = pd.DataFrame(data)
    t.page = page
    t.order = order
    t.shape = t.df.shape
    return t


@pytest.fixture
def tables():
    return TableList(
        [
            _table([["a", "b"], ["c", "d"]], page=1, order=1),
            _table([["e", "f"]], page=2, order=1),
        ]
    )


@pytest.mark.parametrize(
    "fmt,ext",
    [("csv", ".csv"), ("json", ".json"), ("html", ".html"), ("markdown", ".md")],
)
def test_export_textual_formats(tables, tmp_path, fmt, ext):
    tables.export(str(tmp_path / f"out{ext}"), f=fmt)
    # _write_file emits one file per table: "<root>-page-<n>-table-<m><ext>".
    assert list(tmp_path.glob(f"out-page-*{ext}"))


def test_export_excel(tables, tmp_path):
    out = tmp_path / "out.xlsx"
    tables.export(str(out), f="excel")
    assert out.exists()
    # Both tables land as separate sheets.
    assert set(pd.ExcelFile(out).sheet_names) == {"page-1-table-1", "page-2-table-1"}


def test_export_sqlite(tables, tmp_path):
    out = tmp_path / "out.sqlite"
    tables.export(str(out), f="sqlite")
    assert out.exists()
    conn = sqlite3.connect(str(out))
    try:
        names = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
    finally:
        conn.close()
    assert any("page-1" in n for n in names)


@pytest.mark.parametrize(
    "fmt,ext", [("csv", ".csv"), ("excel", ".xlsx"), ("sqlite", ".sqlite")]
)
def test_export_compress_makes_zip(tables, tmp_path, fmt, ext):
    tables.export(str(tmp_path / f"out{ext}"), f=fmt, compress=True)
    assert (tmp_path / "out.zip").exists()


def test_table_to_excel_roundtrip(tmp_path):
    # Regression: Table.to_excel previously passed encoding= (removed in
    # pandas >= 2) and never closed the writer, so it raised / wrote an
    # empty file. It should now produce a readable workbook.
    t = _table([["x", "y"]])
    out = tmp_path / "t.xlsx"
    t.to_excel(str(out))
    assert out.exists()
    back = pd.read_excel(out, header=None)
    assert back.iloc[0, 0] == "x"
    assert back.iloc[0, 1] == "y"


def test_table_to_sqlite(tmp_path):
    t = _table([["x", "y"]], page=3, order=2)
    out = tmp_path / "t.sqlite"
    t.to_sqlite(str(out))
    conn = sqlite3.connect(str(out))
    try:
        rows = conn.execute('SELECT * FROM "page-3-table-2"').fetchall()
    finally:
        conn.close()
    assert rows == [("x", "y")]
