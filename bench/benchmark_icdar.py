"""Benchmark Camelot on the ICDAR-2013 table set already shipped in-repo.

Not a pytest test — run directly with ``python bench/benchmark_icdar.py``.

The dataset (67 born-digital PDFs + structure ground truth) lives under
``tests/files/tabula/icdar2013-dataset/`` and is **not** packaged into the
wheel (only ``camelot/`` ships), so this benchmark is fully self-contained
— no download. It reproduces the F1 / TEDS / row-col numbers used while
tuning the parsers, for any maintainer, from a clean checkout::

    python bench/benchmark_icdar.py --flavor lattice --engine combined
    python bench/benchmark_icdar.py --flavor auto

Independent MIT implementation: the ground-truth parser (ICDAR ``-str.xml``)
and metrics are written from scratch (no AGPL ``tablers-benchmark`` code).
The TEDS here is a fast difflib proxy, not the exact tree-edit metric — fine
for relative comparison between Camelot configurations.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "tests" / "files" / "tabula" / "icdar2013-dataset"


# --------------------------------------------------------------------------
# Ground truth: ICDAR-2013 -str.xml -> {page: [grid, ...]}
# --------------------------------------------------------------------------
def parse_icdar_str_xml(xml_path: Path) -> dict[int, list[list[list[str]]]]:
    """Parse an ICDAR ``-str.xml`` into ``{page: [grid, ...]}``.

    Each ``<region page=N>`` is one table; its ``<cell>`` elements carry
    ``start-row`` / ``start-col`` (and optional ``end-*`` spans) plus
    ``<content>``. We place each cell's text at its top-left grid position
    (spans left blank) — a stable proxy for content comparison.
    """
    out: dict[int, list[list[list[str]]]] = {}
    # noqa justified: input is our own committed ICDAR GT fixtures, not untrusted.
    root = ET.parse(xml_path).getroot()  # noqa: S314
    for region in root.iter("region"):
        page = int(region.get("page", "1"))
        placed = []  # (row, col, text)
        max_r = max_c = 0
        for cell in region.findall("cell"):
            r = int(cell.get("start-row", "0"))
            c = int(cell.get("start-col", "0"))
            content = cell.find("content")
            text = (content.text or "") if content is not None else ""
            placed.append((r, c, " ".join(text.split())))
            max_r, max_c = max(max_r, r), max(max_c, c)
        grid = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
        for r, c, text in placed:
            grid[r][c] = text
        out.setdefault(page, []).append(grid)
    return out


def load_icdar(data_dir: Path, limit: int | None):
    """Yield ``(pdf_path, gt_per_page)`` for each ICDAR doc with GT."""
    n = 0
    for xml in sorted(data_dir.rglob("*-str.xml")):
        pdf = xml.with_name(xml.name.replace("-str.xml", ".pdf"))
        if not pdf.exists():
            continue
        yield pdf, parse_icdar_str_xml(xml)
        n += 1
        if limit and n >= limit:
            return


# --------------------------------------------------------------------------
# Metrics (independent, MIT)
# --------------------------------------------------------------------------
def _seq(grid):
    return [" ".join(str(c or "").split()).lower() for row in grid for c in row]


def simple_teds(pred, gt) -> float:
    """Edit-distance ratio over the row-major cell-text sequence, shape-penalised."""
    p, g = _seq(pred), _seq(gt)
    if not p and not g:
        return 1.0
    content = difflib.SequenceMatcher(None, p, g).ratio()
    pr, pc = len(pred), max((len(r) for r in pred), default=0)
    gr, gc = len(gt), max((len(r) for r in gt), default=0)
    shape = 1.0 - (abs(pr - gr) + abs(pc - gc)) / (pr + pc + gr + gc + 1)
    return content * shape


def score(pred_per_doc, gt_per_doc):
    """Aggregate detection-F1, TEDS, and exact row/col-count accuracy."""
    tp = fp = fn = 0
    teds, rows_ok, cols_ok, matched = [], 0, 0, 0
    for key, gt_pages in gt_per_doc.items():
        pred_pages = pred_per_doc.get(key, {})
        pages = set(gt_pages) | set(pred_pages)
        for pg in pages:
            gts = gt_pages.get(pg, [])
            preds = pred_pages.get(pg, [])
            tp += min(len(gts), len(preds))
            fp += max(0, len(preds) - len(gts))
            fn += max(0, len(gts) - len(preds))
            for gtab, ptab in zip(gts, preds, strict=False):  # match by order
                matched += 1
                teds.append(simple_teds(ptab, gtab))
                rows_ok += len(ptab) == len(gtab)
                cols_ok += max((len(r) for r in ptab), default=0) == max(
                    (len(r) for r in gtab), default=0
                )
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "f1": f1,
        "teds": sum(teds) / len(teds) if teds else 0.0,
        "row": rows_ok / matched if matched else 0.0,
        "col": cols_ok / matched if matched else 0.0,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def _camelot_grids(pdf_path, flavor, engine):
    import camelot

    kwargs = {"pages": "all", "flavor": flavor, "suppress_stdout": True}
    if flavor == "lattice" and engine:
        kwargs["engine"] = engine
    per_page: dict[int, list] = {}
    for t in camelot.read_pdf(str(pdf_path), **kwargs):
        per_page.setdefault(t.page, []).append(t.df.values.tolist())
    return per_page


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--flavor",
        default="lattice",
        choices=["lattice", "stream", "network", "hybrid", "auto"],
    )
    ap.add_argument("--engine", default="combined", help="lattice engine")
    args = ap.parse_args(argv)

    gt_per_doc, pred_per_doc, total = {}, {}, 0.0
    n = 0
    for pdf, gt in load_icdar(args.data_dir, args.limit):
        key = str(pdf)
        gt_per_doc[key] = gt
        tic = time.perf_counter()
        try:
            pred_per_doc[key] = _camelot_grids(pdf, args.flavor, args.engine)
        except Exception as exc:  # noqa: BLE001 - bench records, never crashes
            pred_per_doc[key] = {}
            print(f"  [warn] {pdf.name}: {exc}")
        total += time.perf_counter() - tic
        n += 1

    if not n:
        raise SystemExit(f"No ICDAR docs found under {args.data_dir}")
    m = score(pred_per_doc, gt_per_doc)
    tag = f"{args.flavor}{'/' + args.engine if args.flavor == 'lattice' else ''}"
    print(
        f"ICDAR-2013 [{tag}] docs={n} time={total:.1f}s "
        f"F1={m['f1']:.3f} TEDS={m['teds']:.3f} "
        f"row={m['row']:.3f} col={m['col']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
