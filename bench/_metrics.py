"""Shared metrics for the bench/ table-extraction benchmarks.

Independent MIT implementation (no AGPL ``tablers-benchmark`` code). A fast
difflib-based TEDS *proxy* plus detection-F1 and exact row/col-count
accuracy, so ``benchmark_icdar.py`` and ``benchmark_fintabnet.py`` report
comparable numbers. This is **not** the exact tree-edit TEDS (which needs
``apted``) — it's monotonic and good enough for relative comparison
between camelot configurations.
"""

from __future__ import annotations

import difflib


def _norm(x) -> str:
    return " ".join(str(x or "").split()).lower()


def _seq(grid):
    return [_norm(c) for row in grid for c in row]


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
    """Aggregate detection-F1, TEDS, and exact row/col-count accuracy.

    Both args are ``{doc_key: {page_no: [grid, ...]}}``. Tables on a page
    are matched by order; detection F1 is count-based per page.
    """
    tp = fp = fn = 0
    teds, rows_ok, cols_ok, matched = [], 0, 0, 0
    for key, gt_pages in gt_per_doc.items():
        pred_pages = pred_per_doc.get(key, {})
        for pg in set(gt_pages) | set(pred_pages):
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
