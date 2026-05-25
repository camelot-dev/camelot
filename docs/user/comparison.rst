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

Click any column header to sort. The Camelot column is highlighted; ✓
means "supported out of the box", ✗ "not supported", ◐ "partial /
workaround required".

.. container:: full-width

  .. raw:: html

        <table class="comparison-matrix sortable">
        <thead>
          <tr>
            <th scope="col">Capability</th>
            <th scope="col">Camelot</th>
            <th scope="col">Tabula</th>
            <th scope="col">pdfplumber</th>
            <th scope="col">PyMuPDF</th>
            <th scope="col">gmft</th>
            <th scope="col">unstructured.io</th>
            <th scope="col">tablers</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">License</th>
            <td>MIT</td>
            <td>MIT</td>
            <td>MIT</td>
            <td>AGPL / commercial</td>
            <td>MIT</td>
            <td>Apache 2.0</td>
            <td>MIT</td>
          </tr>
          <tr>
            <th scope="row">Runtime</th>
            <td>pure Python</td>
            <td>Java + wrapper</td>
            <td>pure Python</td>
            <td>C binding</td>
            <td>PyTorch model</td>
            <td>Python + plugins</td>
            <td>Rust + Python</td>
          </tr>
          <tr>
            <th scope="row">Ruled-grid tables</th>
            <td><span class="cm-yes" title="flavor='lattice'">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes" title="model-based">&#10003;</span></td>
            <td><span class="cm-partial" title="via backend">&#9680;</span></td>
            <td><span class="cm-yes" title="line/rect edge detection">&#10003;</span></td>
          </tr>
          <tr>
            <th scope="row">Borderless / whitespace tables</th>
            <td><span class="cm-yes" title="stream / network / hybrid; flavor='ml' for the hardest">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-partial">&#9680;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-no" title="edge-based detection">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Per-page kwarg overrides</th>
            <td><span class="cm-yes" title="per_page= (#41)">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-partial" title="manual loop">&#9680;</span></td>
            <td><span class="cm-partial">&#9680;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Scanned PDFs (no text layer)</th>
            <td><span class="cm-yes" title="flavor='ml' + [ocr] extra">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-partial" title="via OCR plugin">&#9680;</span></td>
            <td><span class="cm-yes" title="vision model">&#10003;</span></td>
            <td><span class="cm-yes" title="Tesseract plugin">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Neural / model-based structure</th>
            <td><span class="cm-yes" title="optional flavor='ml' (Table Transformer)">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-yes" title="Table Transformer">&#10003;</span></td>
            <td><span class="cm-partial" title="via model backends">&#9680;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Confidence score per table</th>
            <td><span class="cm-yes" title="Table.confidence">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-partial" title="heuristic">&#9680;</span></td>
            <td><span class="cm-yes" title="model score">&#10003;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">In-memory bytes / file-like input</th>
            <td><span class="cm-yes" title="#270">&#10003;</span></td>
            <td><span class="cm-no" title="needs path">&#10007;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-yes">&#10003;</span></td>
            <td><span class="cm-no" title="not documented">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Multi-page table stitching</th>
            <td><span class="cm-yes" title="TableList.stack_contiguous (#628)">&#10003;</span></td>
            <td><span class="cm-partial" title="manual">&#9680;</span></td>
            <td><span class="cm-partial">&#9680;</span></td>
            <td><span class="cm-partial">&#9680;</span></td>
            <td><span class="cm-yes" title="model-aware">&#10003;</span></td>
            <td><span class="cm-partial">&#9680;</span></td>
            <td><span class="cm-no">&#10007;</span></td>
          </tr>
          <tr>
            <th scope="row">Heavy native deps</th>
            <td>opencv-headless, pdfium</td>
            <td>JRE</td>
            <td>none</td>
            <td>mupdf (vendored)</td>
            <td>PyTorch (+ GPU)</td>
            <td>varies</td>
            <td>none (Rust/pdfium bundled)</td>
          </tr>
        </tbody>
      </table>

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

* **When gmft wins.** A pure model-first workflow on visually-complex
  tables — bank statements, forms — where you want the neural network
  to drive the whole extraction.
* **When Camelot wins.** Heuristic-first by default (predictable,
  CPU-only, no model weights) — and when you *do* want a model,
  Camelot's optional ``flavor='ml'`` runs the same Table Transformer
  family but fills cell **text from the PDF's own text layer** (or OCR
  for scans) instead of letting the model emit it, so it can't
  hallucinate or alter a value. Plus per-extraction kwargs and a
  per-table ``confidence`` score.
* **Resource cost.** gmft always pulls a Table Transformer checkpoint
  (~hundreds of MB) and benefits from a GPU. Camelot's core needs
  neither; that cost applies only if you opt into ``camelot-py[ml]``.

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

tablers
-------

`tablers <https://github.com/monchin/tablers>`_ is a young, MIT-licensed
extractor with its core algorithms written in **Rust** (exposed to Python
via PyO3) and PDF handling through pdfium — so it installs with no external
Python dependencies and is built for speed.

* **When tablers wins.** Throughput on **ruled** tables — it detects
  tables from line/rectangle edges and is designed to be very fast, with
  lazy page loading for large files. If your PDFs are consistently
  ruled and speed matters, it's worth a look.
* **When Camelot wins.** Breadth: borderless / whitespace tables
  (``stream`` / ``network`` / ``hybrid``), the vector+raster
  ``engine="combined"``, per-table ``accuracy`` / ``whitespace`` /
  ``confidence`` metrics with :meth:`TableList.filter`, multi-page
  stitching, and pandas-DataFrame output. tablers currently focuses on
  edge-detected tables and exports to CSV / Markdown / HTML.

*Last verified: 2026-05-21 against the tablers README (project is new;
assessed from its documented features rather than a benchmark run).*

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
notice. The per-tool prose + capability matrix above are
hand-maintained.

The objective numbers — does each tool run on a given PDF, how many
tables it returns, and how long it takes — are produced by a script,
so they can be refreshed without editing prose::

    $ python bench/comparison.py

That runs Camelot plus every peer extractor that's importable in the
environment (missing ones are skipped, not errored) against a small
canonical corpus, and writes ``docs/_static/comparison_bench.csv``.
The script measures table-count + timing only, not extraction
*quality* (which needs per-PDF ground truth — a separate effort).
Wiring it into a release-time CI job that installs the heavyweight
comparators (a JRE for Tabula, PyTorch for gmft, …) so the CSV
refreshes automatically is the remaining follow-up.

----

For practical recipes that *use* Camelot's specific features —
``per_page``, ``replace_text``, in-memory ``bytes`` input,
``stack_contiguous`` for multi-page tables — see the
:ref:`advanced <advanced>` page.
