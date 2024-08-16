.. Camelot documentation master file, created by
   sphinx-quickstart on Tue Jul 19 13:44:18 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pypdf_table_extraction (Camelot): PDF Table Extraction for Humans
=================================================================


Release v\ |version|. (:ref:`Installation <install>`)

.. image:: https://travis-ci.org/camelot-dev/camelot.svg?branch=master
    :target: https://travis-ci.org/camelot-dev/camelot

.. image:: https://readthedocs.org/projects/camelot-py/badge/?version=master
    :target: https://camelot-py.readthedocs.io/en/master/
    :alt: Documentation Status

.. image:: https://codecov.io/github/py-pdf/pypdf_table_extraction/badge.svg?branch=main&service=github
    :target: https://codecov.io/github/py-pdf/pypdf_table_extraction/?branch=main

.. image:: https://img.shields.io/pypi/v/pypdf-table-extraction.svg
    :target: https://pypi.org/project/pypdf-table-extraction/

.. image:: https://img.shields.io/pypi/l/pypdf-table-extraction.svg
    :target: https://pypi.org/project/pypdf-table-extraction/

.. image:: https://img.shields.io/pypi/pyversions/pypdf-table-extraction.svg
    :target: (https://pypi.org/project/pypdf-table-extraction/


**pypdf_table_extraction** Formerly known as Camelot is a Python library that can help you extract tables from PDFs!

.. note:: You can also check out `Excalibur`_, the web interface to pypdf_table_extraction (Camelot)!

.. _Excalibur: https://github.com/camelot-dev/excalibur

----

**Here's how you can extract tables from PDFs.** You can check out the PDF used in this example `here`_.

.. _here: _static/pdf/foo.pdf

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

pypdf_table_extraction also comes packaged with a :ref:`command-line interface <cli>`!

.. note:: pypdf_table_extraction only works with text-based PDFs and not scanned documents. (As Tabula `explains`_, "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

You can check out some frequently asked questions :ref:`here <faq>`.

.. _explains: https://github.com/tabulapdf/tabula#why-tabula

Why Camelot?
------------

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
