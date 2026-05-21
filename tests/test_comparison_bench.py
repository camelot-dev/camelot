"""Unit tests for bench/comparison.py — the pure (non-tool-running) parts.

The per-tool runners need the real libraries + PDFs, so these tests exercise
the harness logic with mock runners: graceful-skip on a missing module, the
Result shape on success / error, and CSV serialisation. No comparator and no
PDF parse required.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

# bench/ is not a package; load comparison.py by path.
_BENCH = Path(__file__).resolve().parent.parent / "bench" / "comparison.py"
_spec = importlib.util.spec_from_file_location("comparison_bench", _BENCH)
comparison = importlib.util.module_from_spec(_spec)
sys.modules["comparison_bench"] = comparison
_spec.loader.exec_module(comparison)


def test_missing_module_is_skipped_not_errored():
    r = comparison.measure(
        "ghost", "a_module_that_does_not_exist_xyz", lambda p: 1 / 0, "x.pdf"
    )
    assert r.available is False
    assert r.n_tables is None and r.seconds is None and r.error == ""


def test_successful_run_records_count_and_time():
    # 'csv' is always importable -> available=True; runner returns a count.
    r = comparison.measure("fake", "csv", lambda p: 3, "doc.pdf")
    assert r.available is True
    assert r.n_tables == 3
    assert r.seconds is not None and r.seconds >= 0


def test_runner_exception_recorded_as_error_row():
    def boom(_):
        raise RuntimeError("kaboom")

    r = comparison.measure("fake", "csv", boom, "doc.pdf")
    assert r.available is True
    assert r.n_tables is None
    assert "kaboom" in r.error


def test_run_bench_iterates_corpus_x_runners():
    corpus = ["tests/files/foo.pdf", "tests/files/health.pdf"]
    runners = {
        "a": ("csv", lambda p: 1),
        "b": ("csv", lambda p: 2),
    }
    results = comparison.run_bench(corpus=corpus, runners=runners)
    assert len(results) == 4  # 2 pdfs x 2 tools
    assert {r.tool for r in results} == {"a", "b"}


def test_write_csv_shape(tmp_path):
    results = [
        comparison.Result("camelot", "tests/files/foo.pdf", True, 1, 0.123),
        comparison.Result("ghost", "tests/files/foo.pdf", False, None, None),
        comparison.Result(
            "broken", "tests/files/foo.pdf", True, None, None, error="nope"
        ),
    ]
    out = tmp_path / "sub" / "bench.csv"
    comparison.write_csv(results, str(out))
    assert out.exists()  # parent dir auto-created
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["tool", "pdf", "available", "tables", "seconds", "error"]
    # camelot row: available yes, count 1, time formatted to 3dp
    assert rows[1] == ["camelot", "tests/files/foo.pdf", "yes", "1", "0.123", ""]
    # ghost row: skipped -> blanks
    assert rows[2] == ["ghost", "tests/files/foo.pdf", "no", "", "", ""]
    # broken row: error captured
    assert rows[3] == ["broken", "tests/files/foo.pdf", "yes", "", "", "nope"]


def test_registry_has_camelot():
    assert "camelot" in comparison.RUNNERS
    probe, runner = comparison.RUNNERS["camelot"]
    assert probe == "camelot" and callable(runner)


def test_main_writes_csv(tmp_path, monkeypatch):
    # Force every tool to look unavailable so main() runs without comparators.
    monkeypatch.setattr(comparison, "_module_available", lambda name: False)
    out = tmp_path / "out.csv"
    rc = comparison.main([str(out)])
    assert rc == 0
    assert out.exists()
    rows = list(csv.reader(out.open()))
    # header + (len(CORPUS) * len(RUNNERS)) skipped rows
    assert len(rows) == 1 + len(comparison.CORPUS) * len(comparison.RUNNERS)
