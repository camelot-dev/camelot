.. _migration:

Migrating to 2.0
================

Camelot 2.0 rolls up a backend migration, performance work, an optional
neural backend, and a few small breaking changes. For most users ``import
camelot`` and ``camelot.read_pdf(...)`` keep working unchanged. This page
lists what to check when upgrading from 1.0.x, then points at the new
opt-in features.

The full, dated list of changes is in the
`changelog <https://github.com/camelot-dev/camelot/blob/master/CHANGELOG.md>`_.

Breaking changes
----------------

Python 3.10+
~~~~~~~~~~~~~

Python 3.9 (EOL October 2025) is no longer supported. The minimum is now
**Python 3.10**.

``line_scale`` default is 15 (was documented as 40)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CLI and ``read_pdf`` docstring used to *say* the ``flavor="lattice"``
default ``line_scale`` was 40, but the Lattice parser always defaulted to
15. The docs now match the implementation. If you relied on the
documented-but-unimplemented 40, set it explicitly:

.. code-block:: python

   camelot.read_pdf("file.pdf", flavor="lattice", line_scale=40)

``Table.to_excel`` drops the index/header by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Table.to_excel`` now defaults to ``index=False, header=False`` to match
``Table.to_csv`` — Excel exports no longer carry the pandas auto-generated
row index / column header. Opt back in with:

.. code-block:: python

   table.to_excel("out.xlsx", index=True, header=True)

``TableList`` materialises its input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TableList(...)`` now consumes an iterable into a list at construction (so
``bool()`` / ``len()`` work on ``TableList(generator())``). A generator
passed in is exhausted immediately rather than at first access.

``PDFHandler.pages`` is a property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PDFHandler.pages`` is now a lazily-resolved property (was an attribute).
Reads are unchanged; only code that *set* it after subclassing is affected.

PDF backend is now ``playa-pdf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend moved from ``pypdf`` + ``pdfminer.six`` to
`playa-pdf <https://pypi.org/project/playa-pdf/>`_: a smaller install set,
more accurate encrypted-PDF handling, and faster hot paths. Pure ``import
camelot`` callers should see no API change. ``pdfminer.six`` is no longer a
direct dependency — ``playa.miner`` exposes a PDFMiner-compatible layout
API, so imports through Camelot keep working.

Default lattice ``engine`` is ``"combined"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``flavor="lattice"`` now defaults to ``engine="combined"`` (raster OpenCV
detection **plus** the PDF's native vector ruled lines). It is safe by
construction — raster always runs and vector lines can only add — so it is
never worse than the old ``"raster"`` default. Pass ``engine="raster"`` to
restore the exact pre-2.0 behaviour. (There is no ``engine="auto"``.)

New (opt-in) features
---------------------

Neural backend for borderless / scanned tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new optional ``flavor="ml"`` uses a Table Transformer model for table
**structure** and fills cell **text** from the PDF (so it can't hallucinate
values). It targets borderless tables, where the heuristic parsers plateau.
With OCR it also reads **scanned / image-only** PDFs. These pull heavier
dependencies, imported lazily, so the core install is unaffected:

.. code-block:: bash

   pip install "camelot-py[ml]"       # borderless
   pip install "camelot-py[ml,ocr]"   # + scanned PDFs

.. code-block:: python

   tables = camelot.read_pdf("report.pdf", flavor="ml")  # device="cuda"/"xpu" optional

See :ref:`how_it_works` for the design and a borderless benchmark, and
:ref:`comparison` for how Camelot's flavors line up against other tools.

Other additions worth knowing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``flavor="auto"`` picks ``lattice`` or ``network`` per page.
- ``TableList.filter(...)`` drops low-quality tables by row/column count,
  accuracy, or whitespace.
- ``Table.confidence`` — a unified ``[0, 1]`` quality score in
  ``parsing_report``.
- ``per_page=`` overrides, ``replace_text=``, list-form ``strip_text=``,
  ``bytes`` / file-like ``read_pdf`` input, and a ``cpu_count`` cap for
  parallel runs.
