"""Benchmark camelot on FinTabNet — born-digital, borderless-heavy.

Not a pytest test — run directly with ``python bench/benchmark_fintabnet.py``.

Why this exists
---------------
ICDAR-2013 (``benchmark_icdar.py``) is ruled-heavy, so it barely exercises
the ``network`` / ``stream`` (borderless) parsers. FinTabNet is born-digital
**financial** PDFs whose tables are overwhelmingly borderless — the network
parser's home turf — with per-table cell-structure ground truth, so it's the
right place to measure those.

Independent MIT implementation (no AGPL ``tablers-benchmark`` code); metrics
come from ``bench/_metrics.py``.

Two ground-truth formats (the dataset is NOT committed — multi-hundred-MB)
----------------------------------------------------------------------------
**Primary — FinTabNet.c** (the format actually obtainable today). The IBM
source below is currently unreachable, so GT + PDFs come from two HF repos:

    # GT: per-page cell annotations (row/col spans + text)
    curl -L -o fintabnet_c_anno.tar.gz \
      https://huggingface.co/datasets/bsmock/FinTabNet.c/resolve/main/FinTabNet.c-PDF_Annotations.tar.gz
    # PDFs: born-digital pages (paired by filename stem company_year_page_N)
    python -c "from huggingface_hub import snapshot_download as s; \
      s('corvicai/FinTabNet_ComTQA', repo_type='dataset', \
        allow_patterns=['pdfs/*.pdf'], local_dir='comtqa')"

    python bench/benchmark_fintabnet.py \
        --annotations fintabnet_c_anno.tar.gz --pdf-dir comtqa \
        --flavor network --limit 250

**Shim — original IBM FinTabNet JSONL** (``dax-cdn`` source; PubTabNet-style
``html`` per table). Kept for if/when that source returns::

    python bench/benchmark_fintabnet.py --jsonl FinTabNet_1.0.0_table_test.jsonl \
        --data-dir fintabnet/ --flavor network
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # for `import _metrics`
sys.path.insert(0, str(ROOT.parent))  # for `import camelot`

from _metrics import score  # noqa: E402


# --------------------------------------------------------------------------
# Ground-truth builders
# --------------------------------------------------------------------------
def _gt_cells_to_grid(tables_json) -> list[list[list[str]]]:
    """FinTabNet.c per-page JSON -> one cell-text grid per table.

    Each table's ``cells`` carry ``row_nums`` / ``column_nums`` (lists, so
    spans are explicit) and ``json_text_content``. Cell text is placed at
    the span's top-left; spanned positions stay blank (a stable proxy).
    """
    grids = []
    for tbl in tables_json:
        cells = tbl.get("cells", [])
        rs = [c["row_nums"] for c in cells if c.get("row_nums")]
        cs = [c["column_nums"] for c in cells if c.get("column_nums")]
        if not rs or not cs:
            continue
        mr = max(max(r) for r in rs)
        mc = max(max(c) for c in cs)
        grid = [["" for _ in range(mc + 1)] for _ in range(mr + 1)]
        for c in cells:
            if c.get("row_nums") and c.get("column_nums"):
                grid[min(c["row_nums"])][min(c["column_nums"])] = (
                    c.get("json_text_content", "") or ""
                )
        grids.append(grid)
    return grids


def _gt_html_to_grid(html_obj: dict) -> list[list[str]]:
    """Original-FinTabNet PubTabNet-style ``html`` -> single cell grid (shim).

    ``structure.tokens`` is HTML where a cell is the single token ``<td>``
    or the split form ``<td`` … ``colspan="2"`` … ``>``; ``cells`` holds
    text tokens per ``<td>``. ``colspan`` is expanded; ``rowspan`` is kept
    flat (documented simplification).
    """
    tokens = html_obj.get("structure", {}).get("tokens", [])
    cells = html_obj.get("cells", [])
    rows: list[list[str]] = []
    row: list[str] = []
    state = {"i": 0}

    def emit(colspan):
        i = state["i"]
        text = "".join(cells[i].get("tokens", [])) if i < len(cells) else ""
        state["i"] = i + 1
        row.extend([text] * colspan)

    in_attr_td = False
    colspan = 1
    for tok in tokens:
        if tok == "<tr>":
            row = []
        elif tok == "</tr>":
            rows.append(row)
        elif tok == "<td>":
            emit(1)
        elif tok == "<td":
            in_attr_td, colspan = True, 1
        elif in_attr_td and tok.startswith(' colspan="'):
            colspan = int(tok.split('"')[1])
        elif in_attr_td and tok in (">", "/>"):
            emit(colspan)
            in_attr_td = False
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


# --------------------------------------------------------------------------
# Dataset iteration
# --------------------------------------------------------------------------
def _iter_fintabnet_c(annotations: Path, pdf_dir: Path, limit):
    """Yield ``(pdf_path, [gt_grid, ...])`` pairing corvicai PDFs to FinTabNet.c GT.

    ``annotations`` is the ``*-PDF_Annotations`` directory or its ``.tar.gz``.
    PDFs are matched by filename stem (``company_year_page_N``).
    """
    pdfs = sorted(glob.glob(str(pdf_dir / "**" / "*.pdf"), recursive=True))
    tar = members = anno_dir = prefix = None
    if str(annotations).endswith((".tar.gz", ".tgz")):
        tar = tarfile.open(annotations)
        members = set(tar.getnames())
        prefix = next((m.split("/")[0] for m in members if m.endswith(".json")), "")
    else:
        anno_dir = annotations

    n = 0
    for pdf in pdfs:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        name = f"{stem}_tables.json"
        if tar is not None:
            member = f"{prefix}/{name}"
            if member not in members:
                continue
            data = json.load(tar.extractfile(member))
        else:
            path = anno_dir / name
            if not path.exists():
                continue
            data = json.loads(path.read_text())
        grids = _gt_cells_to_grid(data)
        if not grids:
            continue
        yield Path(pdf), grids
        n += 1
        if limit and n >= limit:
            return


def _iter_fintabnet_jsonl(jsonl: Path, data_dir: Path, split, limit):
    """Yield ``(pdf_path, [gt_grid])`` from the original IBM JSONL (shim)."""
    n = 0
    with open(jsonl, encoding="utf-8") as f:
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
            yield pdf, [grid]
            n += 1
            if limit and n >= limit:
                return


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _camelot_grids(pdf_path, flavor, engine):
    import camelot

    kwargs = {"pages": "1", "flavor": flavor, "suppress_stdout": True}
    if flavor in ("lattice", "hybrid") and engine:
        kwargs["engine"] = engine
    return [t.df.values.tolist() for t in camelot.read_pdf(str(pdf_path), **kwargs)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="FinTabNet borderless benchmark")
    ap.add_argument("--annotations", type=Path, help="FinTabNet.c dir or .tar.gz")
    ap.add_argument("--pdf-dir", type=Path, help="dir of born-digital PDFs (corvicai)")
    ap.add_argument("--jsonl", type=Path, help="original IBM FinTabNet JSONL (shim)")
    ap.add_argument("--data-dir", type=Path, help="root for --jsonl PDFs (shim)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument(
        "--flavor",
        default="network",
        choices=["lattice", "stream", "network", "hybrid", "auto"],
    )
    ap.add_argument("--engine", default="combined", help="lattice engine")
    args = ap.parse_args(argv)

    if args.jsonl:
        pairs = _iter_fintabnet_jsonl(
            args.jsonl, args.data_dir or args.jsonl.parent, args.split, args.limit
        )
    elif args.annotations and args.pdf_dir:
        pairs = _iter_fintabnet_c(args.annotations, args.pdf_dir, args.limit)
    else:
        ap.error("pass --annotations + --pdf-dir (FinTabNet.c) or --jsonl (IBM shim)")

    gt_per, pred_per, total, n = {}, {}, 0.0, 0
    for pdf, gt_grids in pairs:
        key = str(pdf)
        gt_per[key] = {1: gt_grids}
        tic = time.perf_counter()
        try:
            pred_per[key] = {1: _camelot_grids(pdf, args.flavor, args.engine)}
        except Exception as exc:  # noqa: BLE001 - bench records, never crashes
            pred_per[key] = {1: []}
            print(f"  [warn] {pdf.name}: {exc}")
        total += time.perf_counter() - tic
        n += 1

    if not n:
        raise SystemExit("No (pdf, gt) pairs — check --annotations / --pdf-dir paths.")
    m = score(pred_per, gt_per)
    tag = f"{args.flavor}"
    if args.flavor in ("lattice", "hybrid"):
        tag += f"/{args.engine}"
    print(
        f"FinTabNet [{tag}] n={n} time={total:.0f}s "
        f"F1={m['f1']:.3f} TEDS={m['teds']:.3f} row={m['row']:.3f} col={m['col']:.3f}"
    )
    return 0


__all__ = ["_gt_html_to_grid", "_gt_cells_to_grid", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
