.. Camelot documentation master file, created by
   sphinx-quickstart on Tue Jul 19 13:44:18 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Camelot: PDF Table Extraction for Humans
========================================

Release v\ |version|. (:ref:`Installation <install>`)

.. image:: https://travis-ci.org/camelot-dev/camelot.svg?branch=master
    :target: https://travis-ci.org/camelot-dev/camelot

.. image:: https://readthedocs.org/projects/camelot-py/badge/?version=master
    :target: https://camelot-py.readthedocs.io/en/master/
    :alt: Documentation Status

.. image:: https://codecov.io/github/camelot-dev/camelot/badge.svg?branch=master&service=github
    :target: https://codecov.io/github/camelot-dev/camelot?branch=master

.. image:: https://img.shields.io/pypi/v/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/

.. image:: https://img.shields.io/pypi/l/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/

.. image:: https://img.shields.io/pypi/pyversions/camelot-py.svg
    :target: https://pypi.org/project/camelot-py/

.. image:: https://badges.gitter.im/camelot-dev/Lobby.png
    :target: https://gitter.im/camelot-dev/Lobby

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

.. image:: https://img.shields.io/badge/continous%20quality-deepsource-lightgrey
    :target: https://deepsource.io/gh/camelot-dev/camelot/?ref=repository-badge

**Camelot** is a Python library that can help you extract tables from PDFs!

.. note:: You can also check out `Excalibur`_, the web interface to Camelot!

.. _Excalibur: https://github.com/camelot-dev/excalibur

----

**Here's how you can extract tables from PDFs.** You can check out the PDF used in this example `here`_.

.. _here: _static/pdf/foo.pdf

::

    >>> import camelot
    >>> tables = camelot.read_pdf('foo.pdf')
    >>> tables
    <TableList n=1>
    >>> tables.export('foo.csv', f='csv', compress=True) # json, excel, html
    >>> tables[0]
    <Table shape=(7, 7)>
    >>> tables[0].parsing_report
    {
        'accuracy': 99.02,
        'whitespace': 12.24,
        'order': 1,
        'page': 1
    }
    >>> tables[0].to_csv('foo.csv') # to_json, to_excel, to_html
    >>> tables[0].df # get a pandas DataFrame!

.. csv-table::
  :file: _static/csv/foo.csv

Camelot also comes packaged with a :ref:`command-line interface <cli>`!

.. note:: Camelot only works with text-based PDFs and not scanned documents. (As Tabula `explains`_, "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

.. _explains: https://github.com/tabulapdf/tabula#why-tabula

Why Camelot?
------------

- **Configurability**: Camelot gives you control over the table extraction process with its :ref:`tweakable settings <advanced>`.
- **Metrics**: Bad tables can be discarded based on metrics like accuracy and whitespace, without having to manually look at each table.
- **Output**: Each table is extracted into a **pandas DataFrame**, which seamlessly integrates into `ETL and data analysis workflows`_. You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML and Sqlite.

.. _ETL and data analysis workflows: https://gist.github.com/vinayak-mehta/e5949f7c2410a0e12f25d3682dc9e873

See `comparison with similar libraries and tools`_.

.. _comparison with similar libraries and tools: https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools

Support the development
-----------------------

If Camelot has helped you, please consider supporting its development with a one-time or monthly donation `on OpenCollective`_!

.. _on OpenCollective: https://opencollective.com/camelot

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
