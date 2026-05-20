.. _comparison:

How Camelot compares to other tools
====================================

This page compares Camelot to the most common open-source PDF
table-extraction libraries you're likely to evaluate alongside it. The
goal isn't to claim Camelot is best for every PDF — different tools win
on different inputs — it's to help you pick the right tool for your
corpus quickly.

If you've already used one of the libraries below, the per-tool
sections name the failure modes that drove Camelot's design choices,
plus the kwargs you can reach for when Camelot's defaults don't fit
your PDFs.

.. note::

    This page was ported from the old GitHub wiki in 2026 and
    refreshed against current releases. Each per-tool section ends
    with a ``Last verified: YYYY-MM-DD`` footer — please open an
    issue if you find an entry has drifted out of date.

At a glance
-----------

.. list-table:: Capability matrix
   :header-rows: 1
   :widths: 18 12 14 14 14 14 14
   :class: full-width

   * - Capability
     - **Camelot**
     - Tabula
     - pdfplumber
     - PyMuPDF
     - gmft
     - unstructured.io
   * - License
     - MIT
     - MIT
     - MIT
     - AGPL / commercial
     - MIT
     - Apache 2.0
   * - Runtime
     - pure Python
     - Java + Python wrapper
     - pure Python
     - C / Python binding
     - PyTorch model
     - Python (pluggable backends)
   * - Ruled-grid tables (lattice)
     - yes (``flavor='lattice'``)
     - yes
     - yes
     - yes
     - yes (model-based)
     - via backend
   * - Borderless / whitespace-separated tables
     - yes (``flavor='stream'``, ``'network'``, ``'hybrid'``)
     - yes
     - partial
     - yes
     - yes
     - yes
   * - Per-page parameter overrides
     - yes (``per_page=``, #41)
     - no
     - manual loop
     - manual loop
     - no
     - no
   * - Scanned PDFs (image-only)
     - no — preprocess with ocrmypdf
     - no
     - no
     - via OCR plugin
     - **yes** (vision model)
     - **yes** (Tesseract / OCR plugin)
   * - Confidence score per table
     - yes (``Table.confidence``)
     - no
     - no
     - heuristic
     - model score
     - no
   * - In-process bytes / file-like input
     - yes (#270)
     - no — needs path
     - yes
     - yes
     - yes
     - yes
   * - Multi-page table stitching
     - yes (``TableList.stack_contiguous``, #628)
     - manual
     - manual
     - manual
     - model-aware
     - manual
   * - Heavy native deps
     - opencv-python-headless, pdfium
     - JRE
     - none
     - mupdf C lib (vendored)
     - PyTorch + CUDA optional
     - varies by backend

Side-by-side example
--------------------

Picking a representative case — `agstat.pdf
<https://github.com/camelot-dev/camelot/blob/master/docs/benchmark/lattice/agstat/agstat.pdf>`_,
a ruled multi-row-header table from a US Department of Agriculture
report — Camelot and Tabula both detect the table area, but Camelot
picks up the merged-header row correctly without manual hinting:

.. list-table::
   :class: full-width

   * - .. figure:: ../benchmark/lattice/agstat/agstat-table-detection-camelot.png
          :width: 95%

          Camelot ``flavor='lattice'``, default kwargs.
     - .. figure:: ../benchmark/lattice/agstat/agstat-table-detection-tabula.png
          :width: 95%

          Tabula auto-detect with the same PDF.

For a quick view of how each tool's CSV output differs on the same
PDF, the `docs/benchmark/
<https://github.com/camelot-dev/camelot/tree/master/docs/benchmark>`_
directory has per-tool CSVs alongside the source PDF for a dozen
test cases (lattice + stream).

Tabula
------

`Tabula <https://github.com/tabulapdf/tabula>`_ is the most direct
peer to Camelot — Camelot's ``flavor='lattice'`` / ``'stream'``
naming is in fact borrowed from Tabula. Tabula ships as a Java
library plus a Python wrapper (`tabula-py
<https://github.com/chezou/tabula-py>`_); the JVM dependency is the
biggest difference for deployment.

* **When Tabula wins.** Auto-detection of stream-flavor tables is
  generally stronger than Camelot's *stream* parser — though
  Camelot's *network* and *hybrid* flavors (added in 1.0) close
  most of the gap on borderless tables. Tabula's interactive web UI
  for manually marking table regions is also unique.
* **When Camelot wins.** Multi-row column headers, merged spanning
  cells, and tables containing italic/superscript decorations.
  Camelot's ``copy_text``, ``shift_text``, ``flag_size``, and
  ``replace_text`` kwargs let you fix specific extraction defects
  without leaving Python.
* **Deployment.** Camelot's pure-Python stack runs in any container
  that has ``opencv-python-headless`` and ``pdfium``; Tabula needs a
  JRE.

*Last verified: 2026-05-21 against tabula-java 1.0.5 / tabula-py 2.10.*

pdfplumber
----------

`pdfplumber <https://github.com/jsvine/pdfplumber>`_ is a layout-
analysis library that grew table-extraction features over time. It's
built on `pdfminer.six` — the same backend Camelot used pre-2.0.

* **When pdfplumber wins.** When you want fine-grained access to
  *every* layout primitive (characters, rects, curves), not just the
  finished table. Pdfplumber exposes the raw layout objects directly,
  making it the right pick for "I want to find tables *and* the
  paragraph headers next to them".
* **When Camelot wins.** Out-of-the-box table-detection quality on
  the typical PDF report; per-table quality reports
  (``parsing_report`` with ``confidence``); the ``flavor='hybrid'``
  parser combining lattice + network signals.
* **Backend.** Camelot has moved past pdfminer.six to `playa-pdf
  <https://pypi.org/project/playa-pdf/>`_ for speed and encrypted-PDF
  correctness; pdfplumber still tracks pdfminer.six.

*Last verified: 2026-05-21 against pdfplumber 0.11.5.*

PyMuPDF (built-in tables)
--------------------------

`PyMuPDF <https://github.com/pymupdf/PyMuPDF>`_ added a
``Page.find_tables()`` API in version 1.23 (2023). It's now a
serious table-extractor backed by the C-level mupdf library.

* **When PyMuPDF wins.** Pure speed on simple ruled tables —
  rasterising is skipped entirely and the C parser is fast. Also a
  good pick if you're already using PyMuPDF for other PDF tasks
  (rendering, text search) and want to keep one dependency.
* **When Camelot wins.** Stream / network / hybrid flavors for
  borderless tables (PyMuPDF's table strategy is geometry-only);
  per-page parameter overrides; multi-page stitching helper.
* **License nuance.** PyMuPDF is AGPL — pulls open-source obligations
  into derivative work unless you buy a commercial licence. Camelot
  is MIT.

*Last verified: 2026-05-21 against PyMuPDF 1.24.x.*

gmft
----

`gmft <https://github.com/conjuncts/gmft>`_ — "Give Me The Formatted
Tables" — is a 2024-era tool that runs Microsoft's Table Transformer
neural network for table detection plus structure recognition. A
different shape from the rule-based tools above.

* **When gmft wins.** Visually-complex tables where a human would
  agree "the cell boundaries are kinda fuzzy" — bank statements,
  forms, OCR'd scans. The vision model handles whitespace, merged
  cells, and even some skew. Works on scanned PDFs unchanged.
* **When Camelot wins.** Predictable per-cell behaviour and
  per-extraction kwargs; no GPU / no half-gigabyte model download;
  stable output for typesetter-generated tables (where the vision
  model is sometimes weirdly creative).
* **Resource cost.** First call downloads a Table Transformer
  checkpoint (~400 MB); inference benefits from a GPU. Camelot runs
  on CPU with no model weights.

*Last verified: 2026-05-21 against gmft 0.4.x.*

unstructured.io
---------------

`unstructured <https://github.com/Unstructured-IO/unstructured>`_ is
a document-preprocessing toolkit aimed at the LLM ingestion pipeline
— it parses PDFs (plus DOCX, HTML, etc.) into a stream of typed
elements (Title, NarrativeText, Table, …).

* **When unstructured wins.** Mixed-content documents where a table
  is one element among many and you want all of them in a single
  pipeline. The OCR / image fallback is built-in via plugins.
* **When Camelot wins.** Table-extraction-only workloads where you
  want maximum control over each table's parameters, want a per-
  table ``confidence`` score, or need the table as a pandas
  ``DataFrame`` rather than a Markdown / HTML serialisation.
* **Output.** unstructured returns tables as HTML / text snippets;
  Camelot returns pandas DataFrames + exporters for CSV / Excel /
  JSON / SQLite / Markdown.

*Last verified: 2026-05-21 against unstructured 0.16.x.*

Tools we no longer compare against
-----------------------------------

The earlier wiki page compared Camelot to two more tools that have
since gone dormant; we evaluated current alternatives and dropped
them from this page:

* `pdftables <https://github.com/drj11/pdftables>`_ — last release
  2014, repository archived. Functional but unmaintained; no
  Python 3.10+ wheels.
* `pdf-table-extract <https://github.com/ashima/pdf-table-extract>`_
  — last release 2017, dormant. Useful historical reference for the
  ruled-line / contour-detection approach but no active maintenance.

If either becomes active again, please `open an issue
<https://github.com/camelot-dev/camelot/issues/new>`_ and we'll add
them back.

Keeping this page up-to-date
-----------------------------

Each per-tool section ends with a ``Last verified: YYYY-MM-DD``
marker so drift is visible without having to dig through commit
history. The intent is for one of these to fall out of date — that's
expected — and for a contributor to refresh it via PR when they
notice. A future iteration of this page may add a ``bench/`` script
that runs each comparator on a canonical PDF corpus and emits a
machine-readable matrix; until then, the matrix above is hand-
maintained.

----

For practical recipes that *use* Camelot's specific features —
``per_page``, ``replace_text``, in-memory ``bytes`` input,
``stack_contiguous`` for multi-page tables — see the
:ref:`advanced <advanced>` page.
