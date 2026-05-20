.. _cli:

Command-Line Interface
======================

Camelot comes with a command-line interface.

You can print the help for the interface by typing ``camelot --help`` in your favorite terminal program, as shown below.
Furthermore, you can print the help for each command by typing ``camelot <command> --help``. Try it out!

Running without installing (uvx)
--------------------------------

If you only want to use the CLI ad-hoc, `uvx <https://docs.astral.sh/uv/concepts/tools/>`_ lets you run it without installing Camelot into the current environment::

    $ uvx camelot-py lattice --output tables.csv document.pdf

The ``camelot-py`` console script is an alias for ``camelot`` matching the PyPI package name, so the older ``uvx --from camelot-py camelot …`` invocation also still works.

Format inference
----------------

``--format`` is optional — when omitted, Camelot infers the format from the ``--output`` path's extension. Supported extensions:

================ ===========
Extension        Format
================ ===========
``.csv``         csv
``.xlsx``/``.xls`` excel
``.html``/``.htm`` html
``.json``        json
``.md``/``.markdown`` markdown
``.sqlite``/``.sqlite3``/``.db`` sqlite
================ ===========

So this works::

    $ camelot-py lattice --output tables.xlsx document.pdf
    # equivalent to: camelot-py lattice --format excel --output tables.xlsx document.pdf

Output is a template
--------------------

``--output`` is treated as a *template* — each detected table is written to ``<output_stem>-page-<P>-table-<T>.<ext>``. So ``--output report.csv`` on a document with 2 tables on page 1 and 1 table on page 3 produces ``report-page-1-table-1.csv``, ``report-page-1-table-2.csv``, ``report-page-3-table-1.csv``.

.. click:: camelot.cli:cli
  :prog: camelot
