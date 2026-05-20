"""Microbenchmark + correctness check for ``get_table_index`` vectorisation.

Not a pytest test - run directly with ``python bench/bench_get_table_index.py``.

This script:

* Builds the legacy Python-loop implementation of ``get_table_index`` (copied
  verbatim from the post-#727 master version) for an apples-to-apples
  comparison against the new NumPy implementation in :mod:`camelot.utils`.
* Generates a synthetic 50 rows x 10 cols table plus 200 random text-like
  objects, then asserts the two implementations return bit-identical results.
* Times both versions over many repetitions and prints
  ``old=X ms new=Y ms speedup=Zx``.

The text-like objects use ``__slots__`` so attribute access cost is
representative of PDFMiner ``LTTextLine`` instances.
"""

from __future__ import annotations

import random
import sys
import time
import warnings
from pathlib import Path

# Ensure we import the in-tree camelot package even when run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camelot.core import Table  # noqa: E402
from camelot.utils import calculate_assignment_error  # noqa: E402,F401
from camelot.utils import flag_font_size  # noqa: E402
from camelot.utils import get_table_index as get_table_index_new  # noqa: E402
from camelot.utils import split_textline  # noqa: E402
from camelot.utils import text_strip  # noqa: E402


class FakeTextLine:
    """Minimal stand-in for ``pdfminer.layout.LTTextLine``."""

    __slots__ = ("x0", "y0", "x1", "y1", "_text", "_objs")

    def __init__(self, x0, y0, x1, y1, text="x"):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self._text = text
        self._objs = []

    def get_text(self):  # noqa: D102
        return self._text


# ---------------------------------------------------------------------------
# Legacy implementation (post-#727, pre-vectorisation) for the head-to-head.
# Kept inline so the benchmark survives future changes to the package code.
# ---------------------------------------------------------------------------
def get_table_index_old(  # noqa: C901, D103
    table, t, direction, split_text=False, flag_size=False, strip_text=""
):
    y_mid = (t.y0 + t.y1) / 2.0
    t_x0, t_x1 = t.x0, t.x1
    rows = table.rows
    cols = table.cols

    r_idx, c_idx = -1, -1
    for r, (y_top, y_bot) in enumerate(rows):
        if y_mid >= y_top:
            break
        if y_mid <= y_bot:
            continue
        best_overlap = -1.0
        best_c = -1
        any_hit = False
        for cidx, (c_left, c_right) in enumerate(cols):
            if c_left <= t_x1 and c_right >= t_x0:
                left = t_x0 if c_left <= t_x0 else c_left
                right = t_x1 if c_right >= t_x1 else c_right
                ov = abs(left - right) / abs(c_left - c_right)
                any_hit = True
                if ov > best_overlap:
                    best_overlap = ov
                    best_c = cidx
        if not any_hit:
            text = t.get_text().strip("\n")
            text_range = (t_x0, t_x1)
            col_range = (cols[0][0], cols[-1][1])
            warnings.warn(
                f"{text} {text_range} does not lie in column range {col_range}",
                stacklevel=1,
            )
        r_idx = r
        c_idx = best_c
        break
    if r_idx == -1:
        return [], 1.0

    error = calculate_assignment_error(t, table, r_idx, c_idx)

    if split_text:
        return (
            split_textline(
                table, t, direction, flag_size=flag_size, strip_text=strip_text
            ),
            error,
        )
    if flag_size:
        return [
            (r_idx, c_idx, flag_font_size(t._objs, direction, strip_text=strip_text))
        ], error
    return [(r_idx, c_idx, text_strip(t.get_text(), strip_text))], error


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------
def make_table(n_rows=50, n_cols=10, page_w=600.0, page_h=800.0):
    """Build a Table with ``n_rows`` x ``n_cols`` evenly spaced cells.

    Rows are returned descending by ``y_top`` to match the parser convention.
    """
    row_h = page_h / n_rows
    col_w = page_w / n_cols
    # Descending y_top.
    rows = [(page_h - i * row_h, page_h - (i + 1) * row_h) for i in range(n_rows)]
    cols = [(i * col_w, (i + 1) * col_w) for i in range(n_cols)]
    return Table(cols, rows)


def make_table_overlapping(n_rows=50, n_cols=10, page_w=600.0, page_h=800.0):
    """Build a Table whose rows deliberately overlap each other.

    Used to exercise the bit-identity fallback path that handles
    non-disjoint row partitions (which the production parsers never
    emit but a downstream consumer could construct by hand).
    """
    row_h = page_h / n_rows
    col_w = page_w / n_cols
    overlap = row_h * 0.4
    rows = [
        (page_h - i * row_h + overlap, page_h - (i + 1) * row_h - overlap)
        for i in range(n_rows)
    ]
    cols = [(i * col_w, (i + 1) * col_w) for i in range(n_cols)]
    return Table(cols, rows)


def make_textlines(n, page_w=600.0, page_h=800.0, seed=0xCAFE10):
    """Generate ``n`` random text-like objects across the page."""
    rng = random.Random(seed)  # noqa: S311
    out = []
    for _ in range(n):
        x0 = rng.uniform(-20.0, page_w + 20.0)  # some out-of-range too
        y0 = rng.uniform(-20.0, page_h + 20.0)
        w = rng.uniform(5.0, 60.0)
        h = rng.uniform(5.0, 20.0)
        out.append(FakeTextLine(x0, y0, x0 + w, y0 + h, text="t"))
    return out


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------
def assert_bit_identical(n_tables=8, n_textlines=200, n_rows=50, n_cols=10):
    """Compare old vs new outputs on many random tables/textlines.

    Exercises both the common non-overlapping row layout and the
    fallback path for overlapping rows that the new implementation has
    to handle to stay bit-identical with the previous Python loop.
    """
    rng = random.Random(0xBEEFCAFE)  # noqa: S311
    mismatches = 0
    factories = (("disjoint", make_table), ("overlapping", make_table_overlapping))
    for k in range(n_tables):
        # Vary the page size to exercise different row/col widths.
        page_w = rng.uniform(300.0, 800.0)
        page_h = rng.uniform(400.0, 1100.0)
        for label, factory in factories:
            table_old = factory(n_rows, n_cols, page_w, page_h)
            table_new = factory(n_rows, n_cols, page_w, page_h)
            textlines = make_textlines(n_textlines, page_w, page_h, seed=0xCAFE10 + k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for t in textlines:
                    old = get_table_index_old(table_old, t, "horizontal")
                    new = get_table_index_new(table_new, t, "horizontal")
                    if old != new:
                        mismatches += 1
                        if mismatches < 5:
                            print(
                                f"MISMATCH [{label}] table={k} "
                                f"t=({t.x0:.2f},{t.y0:.2f},"
                                f"{t.x1:.2f},{t.y1:.2f})"
                                f"\n  old={old}\n  new={new}"
                            )
    if mismatches:
        raise AssertionError(f"{mismatches} mismatches between old and new")
    print(
        f"correctness: OK ({n_tables} tables x {n_textlines} textlines "
        f"x {len(factories)} row layouts)"
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench(n_rows=50, n_cols=10, n_textlines=200, reps=500):
    """Time both implementations head-to-head."""
    table_old = make_table(n_rows, n_cols)
    table_new = make_table(n_rows, n_cols)
    # Warm the lazy numpy cache so we measure only the steady-state cost.
    _ = table_new._rows_np
    _ = table_new._cols_np
    textlines = make_textlines(n_textlines)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Warm-up
        for t in textlines:
            get_table_index_old(table_old, t, "horizontal")
            get_table_index_new(table_new, t, "horizontal")

        t0 = time.perf_counter()
        for _ in range(reps):
            for t in textlines:
                get_table_index_old(table_old, t, "horizontal")
        t1 = time.perf_counter()
        for _ in range(reps):
            for t in textlines:
                get_table_index_new(table_new, t, "horizontal")
        t2 = time.perf_counter()

    old_ms = (t1 - t0) * 1000.0
    new_ms = (t2 - t1) * 1000.0
    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    print(
        f"rows={n_rows} cols={n_cols} textlines={n_textlines} reps={reps} "
        f"-> old={old_ms:.1f} ms  new={new_ms:.1f} ms  "
        f"speedup={speedup:.2f}x"
    )
    return old_ms, new_ms, speedup


def main():  # noqa: D103
    assert_bit_identical()
    # Headline numbers: the size used by the perf report in PR #727
    # (50 rows x 10 cols x 200 textlines x 500 reps).
    bench()
    # Scaling sweep - the vectorised row search shines as n_rows grows.
    print("scaling (rows varied, cols=10, textlines=200, reps=200):")
    for n_rows in (10, 25, 50, 100, 200, 500):
        bench(n_rows=n_rows, reps=200)


if __name__ == "__main__":
    main()
