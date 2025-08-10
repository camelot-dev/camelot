.. documentation master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Camelot: PDF Table Extraction for Humans
========================================

Release v\ |version|. (:ref:`Installation <install>`)

.. image:: https://readthedocs.org/projects/camelot-py/badge/?version=latest
    :target: https://camelot-py.readthedocs.io/
    :alt: Documentation Status

.. image:: https://codecov.io/github/camelot-dev/camelot/badge.svg?branch=master&service=github
    :target: https://codecov.io/github/camelot-dev/camelot/?branch=master

.. image:: https://img.shields.io/pypi/v/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/

.. image:: https://img.shields.io/pypi/l/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/

.. image:: https://img.shields.io/pypi/pyversions/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/


**Camelot** is a Python library that can help you extract tables from PDFs.

.. note:: You can also check out `Excalibur`_, the web interface to Camelot.

.. _Excalibur: https://github.com/camelot-dev/excalibur

----

**Extract tables from PDFs in just a few lines of code:**

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/camelot-dev/camelot/blob/master/examples/camelot-quickstart-notebook.ipynb

Try it yourself in our interactive quickstart notebook. |colab|

Or check out a simple example using `this pdf`_.

.. _this pdf: _static/pdf/foo.pdf

.. code-block:: pycon

    >>> import camelot
    >>> tables = camelot.read_pdf('foo.pdf')
    >>> tables
    <TableList n=1>
    >>> tables.export('foo.csv', f='csv', compress=True) # json, excel, html, markdown, sqlite
    >>> tables[0]
    <Table shape=(7, 7)>
    >>> tables[0].parsing_report
    {
        'accuracy': 99.02,
        'whitespace': 12.24,
        'order': 1,
        'page': 1
    }
    >>> tables[0].to_csv('foo.csv') # to_json, to_excel, to_html, to_markdown, to_sqlite
    >>> tables[0].df # get a pandas DataFrame!

.. csv-table::
  :file: _static/csv/foo.csv

Camelot also comes packaged with a :ref:`command-line interface <cli>`!

.. note:: Camelot only works with text-based PDFs and not scanned documents. (As Tabula `explains`_, "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

You can check out some frequently asked questions :ref:`here <faq>`.

.. _explains: https://github.com/tabulapdf/tabula#why-tabula

Why Camelot?
---------------------------

- **Configurability**: Camelot gives you control over the table extraction process with :ref:`tweakable settings <advanced>`.
- **Metrics**: You can discard bad tables based on metrics like accuracy and whitespace, without having to manually look at each table.
- **Output**: Each table is extracted into a **pandas DataFrame**, which seamlessly integrates into `ETL and data analysis workflows`_. You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML, Markdown, and Sqlite.

.. _ETL and data analysis workflows: https://gist.github.com/vinayak-mehta/e5949f7c2410a0e12f25d3682dc9e873

See `comparison with similar libraries and tools`_.

.. _comparison with similar libraries and tools: https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools


The User Guide
--------------

This part of the documentation begins with some background information about why Camelot was created, takes you through some implementation details, and then focuses on step-by-step instructions for getting the most out of Camelot.

.. toctree::
   :maxdepth: 2

   user/intro
   user/install-deps
   user/install
   user/how-it-works
   user/quickstart
   user/advanced
   user/faq
   user/cli


The API Documentation/Guide
---------------------------

If you are looking for information on a specific function, class, or method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api

The Contributor Guide
---------------------

If you want to contribute to the project, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   dev/contributing
   dev/releasing
   Changelog <https://github.com/camelot-dev/camelot/releases>
