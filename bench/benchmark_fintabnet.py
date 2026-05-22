"""Benchmark Camelot on FinTabNet — a born-digital, borderless-heavy corpus.

Not a pytest test — run directly with ``python bench/benchmark_fintabnet.py``.

Why this exists
---------------
Our ICDAR-2013 runs are *ruled*-heavy, so they barely exercise the
``network`` / ``stream`` (borderless) parsers and can't size changes that
live in that path (e.g. the #619/3 gap-edge formula, ``LAParams``
word-spacing). FinTabNet is born-digital **financial** PDFs whose tables
are overwhelmingly borderless / whitespace-separated — exactly the
network parser's home turf — with per-table HTML structure ground truth,
so it's the right place to measure those.

This is an independent, MIT implementation (Camelot is MIT). It does **not**
reuse code from the AGPL ``tablers-benchmark``; the simple TEDS below is a
small from-scratch difflib approximation of the PubTabNet TEDS idea.

Dataset (one-time, ~ a few GB — NOT committed)
----------------------------------------------
The original IBM FinTabNet bundles the PDFs *and* PubTabNet-style HTML
annotations::

    curl -L -o fintabnet.tar.gz \
      https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/fintabnet.tar.gz
    tar xf fintabnet.tar.gz          # -> fintabnet/  (pdf/  + *.jsonl)

Then point this script at it::

    python bench/benchmark_fintabnet.py --data-dir /path/to/fintabnet \
        --split test --limit 200 --flavor network

Each JSONL row carries ``filename`` (page PDF, relative to the dataset
root) and a PubTabNet-style ``html`` block (``structure.tokens`` +
``cells[].tokens``) which we render to a ground-truth cell grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Ground truth: PubTabNet/FinTabNet HTML -> 2-D cell grid
# --------------------------------------------------------------------------
def _gt_html_to_grid(html_obj: dict) -> list[list[str]]:
    """Render a FinTabNet ``html`` annotation to a dense 2-D text grid.

    ``html_obj`` has ``structure.tokens`` — PubTabNet-style HTML tokens
    where a cell is either the single token ``<td>`` (no attributes) or
    the split form ``<td`` … ``colspan="2"`` … ``>`` — and ``cells`` (text
    tokens per ``<td>`` in document order). ``colspan`` is expanded so the
    grid has one entry per column. ``rowspan`` is **not** expanded
    vertically (this proxy GT is row-major); that's a known simplification,
    acceptable for relative comparisons.
    """
    tokens = html_obj.get("structure", {}).get("tokens", [])
    cells = html_obj.get("cells", [])
    rows: list[list[str]] = []
    row: list[str] = []
    state = {"cell_i": 0}

    def emit(colspan):
        i = state["cell_i"]
        text = "".join(cells[i].get("tokens", [])) if i < len(cells) else ""
        state["cell_i"] = i + 1
        row.extend([text] * colspan)

    in_attr_td = False
    colspan = 1
    for tok in tokens:
        if tok == "<tr>":
            row = []
        elif tok == "</tr>":
            rows.append(row)
        elif tok == "<td>":  # complete, attribute-less opening tag
            emit(1)
        elif tok == "<td":  # opening tag with attributes to follow
            in_attr_td = True
            colspan = 1
        elif in_attr_td and tok.startswith(' colspan="'):
            colspan = int(tok.split('"')[1])
        elif in_attr_td and tok in (">", "/>"):
            emit(colspan)
            in_attr_td = False
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


# --------------------------------------------------------------------------
# Metrics (independent, MIT)
# --------------------------------------------------------------------------
def _norm(s) -> str:
    return " ".join(str(s or "").split()).lower()


def simple_teds(pred: list[list[str]], gt: list[list[str]]) -> float:
    """Cheap TEDS-like score over the row-major cell-text sequence.

    1 - normalised edit distance of the normalised cell texts, times a
    table-shape penalty. This is *not* the exact tree-edit TEDS (which
    needs ``apted``); it's a
    fast, monotonic proxy good enough for relative comparisons between
    Camelot configurations — the same role the field plays in our ICDAR
    runs. Returns a value in ``[0, 1]``.
    """
    import difflib

    pred_seq = [_norm(c) for r in pred for c in r]
    gt_seq = [_norm(c) for r in gt for c in r]
    if not gt_seq and not pred_seq:
        return 1.0
    content = difflib.SequenceMatcher(None, pred_seq, gt_seq).ratio()
    pr, pc = (len(pred), max((len(r) for r in pred), default=0))
    gr, gc = (len(gt), max((len(r) for r in gt), default=0))
    shape = 1.0
    if gr or gc:
        shape = 1.0 - (abs(pr - gr) + abs(pc - gc)) / (gr + gc + pr + pc + 1)
    return content * shape


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def load_fintabnet(data_dir: Path, split: str, limit: int | None):
    """Yield ``(pdf_path, gt_grid)`` for up to ``limit`` ``split`` tables."""
    jsonls = sorted(data_dir.glob("*.jsonl"))
    if not jsonls:
        raise SystemExit(
            f"No *.jsonl found in {data_dir} — see this file's docstring for "
            "the FinTabNet download steps."
        )
    n = 0
    for jl in jsonls:
        with open(jl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if split and rec.get("split") not in (None, split):
                    continue
                pdf = data_dir / rec["filename"]
                if not pdf.exists():
                    continue
                grid = _gt_html_to_grid(rec.get("html", {}))
                if not grid:
                    continue
                yield pdf, grid
                n += 1
                if limit and n >= limit:
                    return


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _camelot_grid(pdf_path, flavor, engine):
    import camelot

    kwargs = {"flavor": flavor, "suppress_stdout": True}
    if flavor == "lattice" and engine:
        kwargs["engine"] = engine
    tables = camelot.read_pdf(str(pdf_path), pages="1", **kwargs)
    if not tables:
        return []
    # FinTabNet pages are cropped to a single table; take the first/largest.
    best = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    return [[c for c in row] for row in best.df.values.tolist()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument(
        "--flavor",
        default="network",
        choices=["lattice", "stream", "network", "hybrid", "auto"],
    )
    ap.add_argument("--engine", default="combined", help="lattice engine")
    args = ap.parse_args(argv)

    scores, times, n = [], 0.0, 0
    for pdf, gt in load_fintabnet(args.data_dir, args.split, args.limit):
        tic = time.perf_counter()
        try:
            pred = _camelot_grid(pdf, args.flavor, args.engine)
        except Exception as exc:  # noqa: BLE001 - bench records, never crashes
            pred = []
            print(f"  [warn] {pdf.name}: {exc}")
        times += time.perf_counter() - tic
        scores.append(simple_teds(pred, gt))
        n += 1

    if not n:
        raise SystemExit("No tables evaluated — check --data-dir / --split.")
    print(
        f"FinTabNet [{args.flavor}"
        f"{'/' + args.engine if args.flavor == 'lattice' else ''}] "
        f"n={n} split={args.split}: simple_TEDS={sum(scores) / n:.3f} "
        f"time={times:.1f}s ({times / n * 1000:.0f} ms/table)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
