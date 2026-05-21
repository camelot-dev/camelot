"""Negative-result microbenchmark for the PERF 4 merge/group/join hotspots.

**This bench is shipped on purpose as a *negative* result.** Every NumPy
candidate it measures is *slower* than the current Python implementation at
the realistic per-page input sizes Camelot actually deals with (n ≤ 500).
The file stays in the tree so anyone future-perf-curious has an immediate
"yes, that was tried, here are the numbers" signal — keep it as a regression
net against well-meaning rewrites.

Not a pytest test - run directly with
``python bench/bench_negative_results.py``.

The perf report flagged these as the next-biggest pure-Python hot loops after
``text_in_bbox`` (PR #731) and ``get_table_index`` (PR #733):

* :func:`camelot.utils.merge_close_lines` (lattice, per page per axis).
* :meth:`camelot.parsers.base.TextBaseParser._group_rows`
  (stream / network, per page).
* :meth:`camelot.parsers.base.TextBaseParser._merge_columns`
  (stream, per page).
* :meth:`camelot.parsers.base.TextBaseParser._join_rows`
  (stream / network, per page).
* :meth:`camelot.parsers.base.TextBaseParser._join_columns`
  (stream, per page).

For each candidate we:

* Freeze a copy of the *legacy* Python-loop implementation in this file so the
  benchmark survives subsequent edits to the package code.
* Provide a candidate NumPy version (``*_np``).
* Assert bit-identical output on random fixtures.
* Time both head-to-head at realistic sizes (``n in (50, 200, 500)``) and
  print ``old=X ms new=Y ms speedup=Zx``.

NumPy dispatch overhead can dominate at small n. PERF 2 (PR #733) found that
``bisect.bisect_left`` beat ``np.searchsorted`` at the per-call scale, so the
expectation is *not* that NumPy automatically helps. We report numbers
honestly so the PR can drop any candidate that does not win.

Findings (run on numpy 2.4.0, CPython 3.13):

* ``merge_close_lines``:        0.4-0.7x  (LOSS - stateful running-mean merge,
                                NumPy dispatch dominates the tiny inner loop).
* ``_group_rows``:              0.8-1.2x  (mixed/no clear win - dominated by
                                ``list.sort`` and per-row ``sorted``, not by
                                the break-detection scan).
* ``_merge_columns``:           0.3-0.5x  (LOSS - stateful overlap merge that
                                cannot be expressed as a simple ``np.diff``).
* ``_join_rows``:               1.0-1.2x  (marginal, and the gain comes from
                                inlining the inner ``max(t.y1 for t in r)``
                                generator, not from NumPy. Pure-Python
                                manual reduce is actually fastest. Out of
                                scope for a NumPy vectorisation task.)
* ``_join_columns``:            0.3-0.4x  (LOSS - the one-shot midpoint
                                comprehension is too small for NumPy to win).

Conclusion: none of these are worth porting at realistic page sizes. The
per-call cost is already sub-millisecond (e.g. ``merge_close_lines`` at
n=500 takes ~0.13 ms per call) and is called once per page per axis. The
PR documents this and keeps the production code unchanged - shipping a
regression is worse than leaving correctness-equivalent code alone.

If a future caller passes ``n > 5000`` (well outside any realistic PDF
page) the NumPy candidates might start to win - but no caller does that
today.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Minimal LTTextLine stand-in (mirrors bench_get_table_index.FakeTextLine).
# ---------------------------------------------------------------------------
class FakeTextLine:
    """Minimal stand-in for ``pdfminer.layout.LTTextLine``."""

    __slots__ = ("x0", "y0", "x1", "y1", "_text")

    def __init__(self, x0, y0, x1, y1, text="x"):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self._text = text

    def get_text(self):  # noqa: D102
        return self._text


# ---------------------------------------------------------------------------
# Legacy implementations (frozen verbatim from upstream master @ ed75b68).
# ---------------------------------------------------------------------------
def merge_close_lines_old(ar, line_tol=2):  # noqa: D103
    ret = []
    for a in ar:
        if not ret:
            ret.append(a)
        else:
            temp = ret[-1]
            if math.isclose(temp, a, abs_tol=line_tol):
                temp = (temp + a) / 2.0
                ret[-1] = temp
            else:
                ret.append(a)
    return ret


def _group_rows_old(text, row_tol=2):  # noqa: D103
    row_y = None
    rows = []
    temp = []
    text.sort(key=lambda x: (-x.y0, x.x0))
    non_empty_text = [t for t in text if t.get_text().strip()]
    for t in non_empty_text:
        if row_y is None:
            row_y = t.y0
        elif not math.isclose(row_y, t.y0, abs_tol=row_tol):
            rows.append(sorted(temp, key=lambda t: t.x0))
            temp = []
            row_y = t.y0
        temp.append(t)
    rows.append(sorted(temp, key=lambda t: t.x0))
    return rows


def _merge_columns_old(cl, column_tol=0):  # noqa: C901, D103
    merged = []
    for higher in cl:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if column_tol >= 0:
                if higher[0] <= lower[1] or math.isclose(
                    higher[0], lower[1], abs_tol=column_tol
                ):
                    upper_bound = max(lower[1], higher[1])
                    lower_bound = min(lower[0], higher[0])
                    merged[-1] = (lower_bound, upper_bound)
                else:
                    merged.append(higher)
            elif column_tol < 0:
                if higher[0] <= lower[1]:
                    if math.isclose(higher[0], lower[1], abs_tol=abs(column_tol)):
                        merged.append(higher)
                    else:
                        upper_bound = max(lower[1], higher[1])
                        lower_bound = min(lower[0], higher[0])
                        merged[-1] = (lower_bound, upper_bound)
                else:
                    merged.append(higher)
    return merged


def _join_rows_old(rows_grouped, text_y_max, text_y_min):  # noqa: D103
    row_boundaries = [
        [max(t.y1 for t in r), min(t.y0 for t in r)] for r in rows_grouped
    ]
    for i in range(0, len(row_boundaries) - 1):
        top_row = row_boundaries[i]
        bottom_row = row_boundaries[i + 1]
        top_row[1] = bottom_row[0] = (top_row[1] + bottom_row[0]) / 2
    row_boundaries[0][0] = text_y_max
    row_boundaries[-1][1] = text_y_min
    return row_boundaries


def _join_columns_old(cols, text_x_min, text_x_max):  # noqa: D103
    cols = sorted(cols)
    cols = [(cols[i][0] + cols[i - 1][1]) / 2 for i in range(1, len(cols))]
    cols.insert(0, text_x_min)
    cols.append(text_x_max)
    cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
    return cols


# ---------------------------------------------------------------------------
# Candidate NumPy implementations.
#
# Each is the best vectorised port the author was able to write while
# staying bit-identical to the legacy semantics. See the module docstring
# for the verdict on each.
# ---------------------------------------------------------------------------
def merge_close_lines_np(ar, line_tol=2):
    """Vectorise the running-mean merge.

    The legacy loop is stateful: each merged element averages with the
    *running mean* of the current group, not with the first element. So a
    plain ``np.diff < tol`` partition is **not** semantically equivalent
    (try ``[0, 1, 2, 3, 4]`` with ``tol=2`` to see the running mean
    absorb everything into a single point).

    The best we can do is a NumPy-checked fast path: if no two adjacent
    inputs are within ``line_tol`` then no merges happen and the legacy
    loop's output is the input list. Otherwise we fall back to the
    legacy loop so semantics stay bit-identical.
    """
    n = len(ar)
    if n < 2:
        return list(ar)
    arr = np.asarray(ar, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    if not np.any(diffs <= line_tol):
        # No two adjacent entries are within tolerance -> no merges happen.
        return [float(x) for x in arr]
    return merge_close_lines_old(ar, line_tol=line_tol)


def _group_rows_np(text, row_tol=2):
    """Vectorise the row-grouping break detection.

    Mirrors the legacy semantics: ``row_y`` is the ``y0`` of the *first*
    element of the current group and stays fixed until the group breaks
    via ``not math.isclose(row_y, y0, abs_tol=row_tol)``.

    Since ``row_y`` resets on each break, the partition is genuinely
    sequential - a true ``np.diff`` shortcut is *not* legal here (gaps
    can ratchet without ever exceeding ``row_tol`` while still drifting
    the cluster mean far from the anchor). We extract ``y0`` to a
    NumPy array up front and build break indices in a tight Python pass.
    """
    text.sort(key=lambda x: (-x.y0, x.x0))
    non_empty_text = [t for t in text if t.get_text().strip()]
    n = len(non_empty_text)
    if n == 0:
        return [[]]
    y0 = np.fromiter((t.y0 for t in non_empty_text), dtype=np.float64, count=n)
    starts = [0]
    anchor = y0[0]
    for i in range(1, n):
        if not math.isclose(anchor, y0[i], abs_tol=row_tol):
            starts.append(i)
            anchor = y0[i]
    starts.append(n)
    rows = []
    for a, b in zip(starts[:-1], starts[1:], strict=False):
        rows.append(sorted(non_empty_text[a:b], key=lambda t: t.x0))
    return rows


def _merge_columns_np(cl, column_tol=0):
    """Vectorise the overlap-or-close column merge.

    Legacy semantics:

    * For ``column_tol >= 0``: merge when ``higher[0] <= lower[1]`` OR
      ``isclose(higher[0], lower[1], abs_tol=column_tol)``.
    * For ``column_tol < 0``: merge when ``higher[0] <= lower[1]`` AND
      NOT ``isclose(higher[0], lower[1], abs_tol=|column_tol|)``.

    The *merged* upper bound is ``max(lower[1], higher[1])``, which can
    swallow many subsequent intervals - so this is stateful in the same
    way as ``merge_close_lines``. We pre-build NumPy arrays for the
    column lows/highs and use a single Python pass for the break logic.
    """
    n = len(cl)
    if n == 0:
        return []
    lows = np.fromiter((c[0] for c in cl), dtype=np.float64, count=n)
    highs = np.fromiter((c[1] for c in cl), dtype=np.float64, count=n)
    merged_lows = [lows[0]]
    merged_highs = [highs[0]]
    if column_tol >= 0:
        for i in range(1, n):
            cur_high = merged_highs[-1]
            if lows[i] <= cur_high or math.isclose(
                lows[i], cur_high, abs_tol=column_tol
            ):
                if highs[i] > cur_high:
                    merged_highs[-1] = highs[i]
                if lows[i] < merged_lows[-1]:
                    merged_lows[-1] = lows[i]
            else:
                merged_lows.append(lows[i])
                merged_highs.append(highs[i])
    else:
        abs_tol = abs(column_tol)
        for i in range(1, n):
            cur_high = merged_highs[-1]
            if lows[i] <= cur_high:
                if math.isclose(lows[i], cur_high, abs_tol=abs_tol):
                    merged_lows.append(lows[i])
                    merged_highs.append(highs[i])
                else:
                    if highs[i] > cur_high:
                        merged_highs[-1] = highs[i]
                    if lows[i] < merged_lows[-1]:
                        merged_lows[-1] = lows[i]
            else:
                merged_lows.append(lows[i])
                merged_highs.append(highs[i])
    return [
        (float(a), float(b)) for a, b in zip(merged_lows, merged_highs, strict=False)
    ]


def _join_rows_np(rows_grouped, text_y_max, text_y_min):
    """Vectorise the per-row min/max + adjacent-row midpoint blend.

    Per-row ``max(t.y1)`` and ``min(t.y0)`` are tight Python generators
    in the legacy version. The midpoint blend
    ``(top[1] + bot[0]) / 2`` over all adjacent row pairs is trivially
    vectorisable; we use ``(y_bot[:-1] + y_top[1:]) / 2`` for it.

    Note: the speedup observed in the bench (~1.0-1.2x) comes mostly
    from inlining the per-row reductions as explicit Python loops, not
    from NumPy. A pure-Python manual-loop version (no NumPy) is in fact
    *faster* than this one at realistic row sizes.
    """
    if not rows_grouped:
        return []
    y_top = np.empty(len(rows_grouped), dtype=np.float64)
    y_bot = np.empty(len(rows_grouped), dtype=np.float64)
    for i, r in enumerate(rows_grouped):
        if not r:
            raise ValueError("empty row group")
        m1 = r[0].y1
        m0 = r[0].y0
        for t in r[1:]:
            if t.y1 > m1:
                m1 = t.y1
            if t.y0 < m0:
                m0 = t.y0
        y_top[i] = m1
        y_bot[i] = m0
    if len(rows_grouped) > 1:
        mids = (y_bot[:-1] + y_top[1:]) / 2.0
        y_bot[:-1] = mids
        y_top[1:] = mids
    y_top[0] = text_y_max
    y_bot[-1] = text_y_min
    return [[float(y_top[i]), float(y_bot[i])] for i in range(len(rows_grouped))]


def _join_columns_np(cols, text_x_min, text_x_max):
    """Vectorise the column-midpoint join.

    Legacy code uses three Python list comprehensions over n cols. The
    midpoint pass ``(cols[i][0] + cols[i-1][1]) / 2`` looks like a
    perfect candidate for ``(lows[1:] + highs[:-1]) / 2.0``, but in
    practice the function is called once per page on a sub-100-element
    list and the ``np.fromiter`` + ndarray allocation overhead dwarfs
    the inner arithmetic.
    """
    n = len(cols)
    if n == 0:
        return [(text_x_min, text_x_max)]
    cols = sorted(cols)
    lows = np.fromiter((c[0] for c in cols), dtype=np.float64, count=n)
    highs = np.fromiter((c[1] for c in cols), dtype=np.float64, count=n)
    if n == 1:
        return [(text_x_min, text_x_max)]
    mids = (lows[1:] + highs[:-1]) / 2.0
    edges = np.empty(n + 1, dtype=np.float64)
    edges[0] = text_x_min
    edges[1:-1] = mids
    edges[-1] = text_x_max
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def make_sorted_floats(n, low=0.0, high=800.0, seed=0):
    """Random sorted floats - lattice-style line coords."""
    rng = random.Random(seed)  # noqa: S311
    return sorted(rng.uniform(low, high) for _ in range(n))


def make_text(n, page_w=600.0, page_h=800.0, seed=0):
    """Random fake textlines for ``_group_rows`` / ``_join_rows``."""
    rng = random.Random(seed)  # noqa: S311
    out = []
    for _ in range(n):
        x0 = rng.uniform(0.0, page_w)
        y0 = rng.uniform(0.0, page_h)
        w = rng.uniform(5.0, 60.0)
        h = rng.uniform(5.0, 20.0)
        out.append(FakeTextLine(x0, y0, x0 + w, y0 + h, text="t"))
    return out


def make_columns(n, page_w=600.0, seed=0):
    """Random (lo, hi) column tuples sorted by lo."""
    rng = random.Random(seed)  # noqa: S311
    cols = []
    for _ in range(n):
        lo = rng.uniform(0.0, page_w)
        hi = lo + rng.uniform(5.0, 60.0)
        cols.append((lo, hi))
    cols.sort()
    return cols


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
def check_merge_close_lines(n_runs=50):
    """Random equivalence check for ``merge_close_lines``."""
    rng = random.Random(0xBEEF1)  # noqa: S311
    for k in range(n_runs):
        n = rng.randint(0, 500)
        ar = make_sorted_floats(n, seed=k)
        for tol in (0.5, 2, 5):
            old = merge_close_lines_old(list(ar), line_tol=tol)
            new = merge_close_lines_np(list(ar), line_tol=tol)
            if old != new:
                raise AssertionError(
                    f"merge_close_lines mismatch n={n} tol={tol}\n  "
                    f"old={old[:5]}...\n  new={new[:5]}..."
                )
    print(f"correctness: merge_close_lines OK ({n_runs} runs)")


def check_group_rows(n_runs=20):
    """Random equivalence check for ``_group_rows``."""
    for k in range(n_runs):
        n = 50 + k * 20
        text = make_text(n, seed=k)
        for tol in (1, 2, 5):
            t_old = [FakeTextLine(t.x0, t.y0, t.x1, t.y1, t._text) for t in text]
            t_new = [FakeTextLine(t.x0, t.y0, t.x1, t.y1, t._text) for t in text]
            old = _group_rows_old(t_old, row_tol=tol)
            new = _group_rows_np(t_new, row_tol=tol)
            if [len(r) for r in old] != [len(r) for r in new]:
                raise AssertionError(
                    f"_group_rows row counts mismatch n={n} tol={tol}\n  "
                    f"old={[len(r) for r in old]}\n  new={[len(r) for r in new]}"
                )
            for ro, rn in zip(old, new, strict=False):
                if [(t.x0, t.y0) for t in ro] != [(t.x0, t.y0) for t in rn]:
                    raise AssertionError(
                        f"_group_rows ordering mismatch n={n} tol={tol}"
                    )
    print(f"correctness: _group_rows OK ({n_runs} runs)")


def check_merge_columns(n_runs=50):
    """Random equivalence check for ``_merge_columns``."""
    rng = random.Random(0xBEEF2)  # noqa: S311
    for k in range(n_runs):
        n = rng.randint(0, 500)
        cols = make_columns(n, seed=k)
        for tol in (-2, 0, 1, 5):
            old = _merge_columns_old(list(cols), column_tol=tol)
            new = _merge_columns_np(list(cols), column_tol=tol)
            if old != new:
                raise AssertionError(
                    f"_merge_columns mismatch n={n} tol={tol}\n  "
                    f"old={old[:5]}...\n  new={new[:5]}..."
                )
    print(f"correctness: _merge_columns OK ({n_runs} runs)")


def check_join_rows(n_runs=10):
    """Random equivalence check for ``_join_rows``."""
    for k in range(n_runs):
        n = 50 + k * 20
        text = make_text(n, seed=k)
        rg = _group_rows_old(list(text), row_tol=2)
        old = _join_rows_old([list(r) for r in rg], text_y_max=900.0, text_y_min=-50.0)
        new = _join_rows_np([list(r) for r in rg], text_y_max=900.0, text_y_min=-50.0)
        if old != new:
            raise AssertionError(
                f"_join_rows mismatch n={n}\n  old={old[:5]}...\n  new={new[:5]}..."
            )
    print(f"correctness: _join_rows OK ({n_runs} runs)")


def check_join_columns(n_runs=50):
    """Random equivalence check for ``_join_columns``."""
    rng = random.Random(0xBEEF3)  # noqa: S311
    for k in range(n_runs):
        n = rng.randint(1, 500)
        cols = make_columns(n, seed=k)
        old = _join_columns_old(list(cols), text_x_min=-10.0, text_x_max=700.0)
        new = _join_columns_np(list(cols), text_x_min=-10.0, text_x_max=700.0)
        if old != new:
            raise AssertionError(
                f"_join_columns mismatch n={n}\n  old={old[:5]}...\n  new={new[:5]}..."
            )
    print(f"correctness: _join_columns OK ({n_runs} runs)")


# ---------------------------------------------------------------------------
# Benchmarks.
# ---------------------------------------------------------------------------
def _time(fn, args, reps):
    fn(*args)
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - t0) * 1000.0


def _print_row(n, reps, old_ms, new_ms):
    sp = old_ms / new_ms if new_ms > 0 else float("inf")
    print(
        f"n={n:>4} reps={reps} -> old={old_ms:7.2f} ms  "
        f"new={new_ms:7.2f} ms  speedup={sp:.2f}x"
    )


def _run_merge_close_lines(ar, tol):
    return merge_close_lines_old(list(ar), tol)


def _run_merge_close_lines_np(ar, tol):
    return merge_close_lines_np(list(ar), tol)


def bench_merge_close_lines(reps=2000):
    print("\n=== merge_close_lines ===")
    for n in (50, 200, 500):
        ar = make_sorted_floats(n, seed=1)
        old_ms = _time(_run_merge_close_lines, (ar, 2), reps)
        new_ms = _time(_run_merge_close_lines_np, (ar, 2), reps)
        _print_row(n, reps, old_ms, new_ms)


def _run_group_rows_old(text):
    return _group_rows_old(
        [FakeTextLine(t.x0, t.y0, t.x1, t.y1, t._text) for t in text], row_tol=2
    )


def _run_group_rows_np(text):
    return _group_rows_np(
        [FakeTextLine(t.x0, t.y0, t.x1, t.y1, t._text) for t in text], row_tol=2
    )


def bench_group_rows(reps=500):
    print("\n=== _group_rows ===")
    for n in (50, 200, 500):
        text = make_text(n, seed=2)
        old_ms = _time(_run_group_rows_old, (text,), reps)
        new_ms = _time(_run_group_rows_np, (text,), reps)
        _print_row(n, reps, old_ms, new_ms)


def _run_merge_columns_old(cols, tol):
    return _merge_columns_old(list(cols), tol)


def _run_merge_columns_np(cols, tol):
    return _merge_columns_np(list(cols), tol)


def bench_merge_columns(reps=2000):
    print("\n=== _merge_columns ===")
    for n in (50, 200, 500):
        cols = make_columns(n, seed=3)
        old_ms = _time(_run_merge_columns_old, (cols, 0), reps)
        new_ms = _time(_run_merge_columns_np, (cols, 0), reps)
        _print_row(n, reps, old_ms, new_ms)


def _run_join_rows_old(rg):
    return _join_rows_old(rg, 900.0, -50.0)


def _run_join_rows_np(rg):
    return _join_rows_np(rg, 900.0, -50.0)


def bench_join_rows(reps=2000):
    print("\n=== _join_rows ===")
    for n in (50, 200, 500):
        text = make_text(n, seed=4)
        rg = _group_rows_old(list(text), row_tol=2)
        rg_lists = [list(r) for r in rg]
        old_ms = _time(_run_join_rows_old, (rg_lists,), reps)
        new_ms = _time(_run_join_rows_np, (rg_lists,), reps)
        _print_row(n, reps, old_ms, new_ms)


def _run_join_columns_old(cols):
    return _join_columns_old(list(cols), -10.0, 700.0)


def _run_join_columns_np(cols):
    return _join_columns_np(list(cols), -10.0, 700.0)


def bench_join_columns(reps=2000):
    print("\n=== _join_columns ===")
    for n in (50, 200, 500):
        cols = make_columns(n, seed=5)
        old_ms = _time(_run_join_columns_old, (cols,), reps)
        new_ms = _time(_run_join_columns_np, (cols,), reps)
        _print_row(n, reps, old_ms, new_ms)


def main():  # noqa: D103
    check_merge_close_lines()
    check_group_rows()
    check_merge_columns()
    check_join_rows()
    check_join_columns()

    bench_merge_close_lines()
    bench_group_rows()
    bench_merge_columns()
    bench_join_rows()
    bench_join_columns()


if __name__ == "__main__":
    main()
