"""Cross-tool comparison bench for the docs comparison page (#31, #763 context).

Not a pytest test — run directly with ``python bench/comparison.py``.

Runs Camelot and whichever peer table-extractors are importable against a
small canonical PDF corpus, and emits a machine-readable CSV
(``docs/_static/comparison_bench.csv``) that ``docs/user/comparison.rst``
can render via ``.. csv-table::``. This is the "Option A" follow-up to
that page's hand-maintained matrix: numbers refresh by re-running the
bench rather than editing prose.

Design notes
------------

* **Graceful degradation.** Each peer tool is behind a guarded import; a
  missing dependency (no Java for Tabula, no PyTorch for gmft, …) is
  recorded as ``available=False`` and skipped, never an error. So the
  bench runs anywhere — CI installs as many comparators as are practical
  and the CSV reflects exactly what was measured.
* **What it measures.** Per (tool, pdf): whether the tool ran, the number
  of tables it returned, and wall-clock seconds. It deliberately does
  *not* score extraction *quality* — that needs ground-truth dataframes
  per PDF and is a separate, larger effort. Table-count + timing is the
  cheap, objective, auto-refreshable signal.
* **CI wiring (follow-up).** A release-time job that pip-installs the
  heavyweight comparators (tabula-py + a JRE, gmft + torch, unstructured)
  and runs this script is the remaining piece; until then the bench runs
  with the pure-Python comparators that are cheap to install
  (pdfplumber, pymupdf), plus Camelot.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Canonical corpus — small, in-tree, covers a ruled (lattice) table, a
#: borderless (stream) table, and a two-table page. Paths are relative to
#: the repo root.
CORPUS = [
    "tests/files/foo.pdf",
    "tests/files/health.pdf",
    "tests/files/tabula/12s0324.pdf",
]


@dataclass
class Result:
    """One (tool, pdf) measurement row."""

    tool: str
    pdf: str
    available: bool
    n_tables: int | None
    seconds: float | None
    error: str = ""


def _module_available(name: str) -> bool:
    """True if ``name`` can be imported without actually importing it."""
    return importlib.util.find_spec(name) is not None


# --- per-tool runners -----------------------------------------------------
#
# Each returns the number of tables found. Raising is fine — the caller
# records it as an error row. Each is paired with an availability probe so
# we never import a missing dependency.


def _run_camelot(pdf_path: str) -> int:
    import camelot

    # Try lattice first, fall back to stream — the bench is about "did the
    # tool get tables", not about flavor selection.
    tables = camelot.read_pdf(pdf_path, flavor="lattice")
    if len(tables) == 0:
        tables = camelot.read_pdf(pdf_path, flavor="stream")
    return len(tables)


def _run_pdfplumber(pdf_path: str) -> int:
    import pdfplumber

    n = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            n += len(page.find_tables())
    return n


def _run_pymupdf(pdf_path: str) -> int:
    import pymupdf  # PyMuPDF >= 1.24 exposes the 'pymupdf' name

    n = 0
    doc = pymupdf.open(pdf_path)
    try:
        for page in doc:
            n += len(page.find_tables().tables)
    finally:
        doc.close()
    return n


#: tool name -> (import-probe module, runner). Add tabula-py / gmft /
#: unstructured here once the CI install matrix supports them.
RUNNERS = {
    "camelot": ("camelot", _run_camelot),
    "pdfplumber": ("pdfplumber", _run_pdfplumber),
    "pymupdf": ("pymupdf", _run_pymupdf),
}


def measure(tool: str, probe_module: str, runner, pdf_path: str) -> Result:
    """Run one tool on one PDF, capturing availability / count / timing."""
    rel = os.path.relpath(pdf_path, ROOT)
    if not _module_available(probe_module):
        return Result(tool, rel, available=False, n_tables=None, seconds=None)
    start = time.perf_counter()
    try:
        n = runner(pdf_path)
    except Exception as exc:  # noqa: BLE001 - bench records, never crashes
        return Result(
            tool, rel, available=True, n_tables=None, seconds=None, error=str(exc)
        )
    return Result(
        tool, rel, available=True, n_tables=n, seconds=time.perf_counter() - start
    )


def run_bench(corpus=None, runners=None) -> list[Result]:
    """Measure every available tool against every PDF in the corpus."""
    corpus = corpus if corpus is not None else CORPUS
    runners = runners if runners is not None else RUNNERS
    results: list[Result] = []
    for rel_pdf in corpus:
        pdf_path = str(ROOT / rel_pdf)
        for tool, (probe, runner) in runners.items():
            results.append(measure(tool, probe, runner, pdf_path))
    return results


def write_csv(results: list[Result], out_path: str) -> None:
    """Write results to ``out_path`` as CSV (header + one row per result)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "pdf", "available", "tables", "seconds", "error"])
        for r in results:
            w.writerow(
                [
                    r.tool,
                    r.pdf,
                    "yes" if r.available else "no",
                    "" if r.n_tables is None else r.n_tables,
                    "" if r.seconds is None else f"{r.seconds:.3f}",
                    r.error,
                ]
            )


def main(argv=None) -> int:
    """Run the bench and write the CSV (default: docs/_static)."""
    argv = argv if argv is not None else sys.argv[1:]
    out = argv[0] if argv else str(ROOT / "docs" / "_static" / "comparison_bench.csv")
    results = run_bench()
    write_csv(results, out)
    for r in results:
        status = (
            "skipped (not installed)"
            if not r.available
            else (
                f"ERROR: {r.error}"
                if r.error
                else f"{r.n_tables} tables in {r.seconds:.3f}s"
            )
        )
        print(f"{r.tool:12} {r.pdf:32} {status}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
