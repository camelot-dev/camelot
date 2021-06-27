.. _quickstart:

Quickstart
==========

In a hurry to extract tables from PDFs? This document gives a good introduction to help you get started with Camelot.

Read the PDF
------------

Reading a PDF to extract tables with Camelot is very simple.

Begin by importing the Camelot module::

    >>> import camelot

Now, let's try to read a PDF. (You can check out the PDF used in this example `here`_.) Since the PDF has a table with clearly demarcated lines, we will use the :ref:`Lattice <lattice>` method here.

.. note:: :ref:`Lattice <lattice>` is used by default. You can use :ref:`Stream <stream>` with ``flavor='stream'``.

.. _here: ../_static/pdf/foo.pdf

::

    >>> tables = camelot.read_pdf('foo.pdf')
    >>> tables
    <TableList n=1>

Now, we have a :class:`TableList <camelot.core.TableList>` object called ``tables``, which is a list of :class:`Table <camelot.core.Table>` objects. We can get everything we need from this object.

We can access each table using its index. From the code snippet above, we can see that the ``tables`` object has only one table, since ``n=1``. Let's access the table using the index ``0`` and take a look at its ``shape``.

::

    >>> tables[0]
    <Table shape=(7, 7)>

Let's print the parsing report.

::

    >>> print tables[0].parsing_report
    {
        'accuracy': 99.02,
        'whitespace': 12.24,
        'order': 1,
        'page': 1
    }

Woah! The accuracy is top-notch and there is less whitespace, which means the table was most likely extracted correctly. You can access the table as a pandas DataFrame by using the :class:`table <camelot.core.Table>` object's ``df`` property.

::

    >>> tables[0].df

.. csv-table::
  :file: ../_static/csv/foo.csv

Looks good! You can now export the table as a CSV file using its :meth:`to_csv() <camelot.core.Table.to_csv>` method. Alternatively you can use :meth:`to_json() <camelot.core.Table.to_json>`, :meth:`to_excel() <camelot.core.Table.to_excel>` :meth:`to_html() <camelot.core.Table.to_html>` :meth:`to_markdown() <camelot.core.Table.to_markdown>` or :meth:`to_sqlite() <camelot.core.Table.to_sqlite>` methods to export the table as JSON, Excel, HTML files or a sqlite database respectively.

::

    >>> tables[0].to_csv('foo.csv')

This will export the table as a CSV file at the path specified. In this case, it is ``foo.csv`` in the current directory.

You can also export all tables at once, using the :class:`tables <camelot.core.TableList>` object's :meth:`export() <camelot.core.TableList.export>` method.

::

    >>> tables.export('foo.csv', f='csv')

.. tip::
    Here's how you can do the same with the :ref:`command-line interface <cli>`.
    ::

        $ camelot --format csv --output foo.csv lattice foo.pdf

This will export all tables as CSV files at the path specified. Alternatively, you can use ``f='json'``, ``f='excel'``, ``f='html'``, ``f='markdown'`` or ``f='sqlite'``.

.. note:: The :meth:`export() <camelot.core.TableList.export>` method exports files with a ``page-*-table-*`` suffix. In the example above, the single table in the list will be exported to ``foo-page-1-table-1.csv``. If the list contains multiple tables, multiple CSV files will be created. To avoid filling up your path with multiple files, you can use ``compress=True``, which will create a single ZIP file at your path with all the CSV files.

.. note:: Camelot handles rotated PDF pages automatically. As an exercise, try to extract the table out of `this PDF`_.

.. _this PDF: ../_static/pdf/rotated.pdf

Specify page numbers
--------------------

By default, Camelot only uses the first page of the PDF to extract tables. To specify multiple pages, you can use the ``pages`` keyword argument::

    >>> camelot.read_pdf('your.pdf', pages='1,2,3')

.. tip::
    Here's how you can do the same with the :ref:`command-line interface <cli>`.
    ::

        $ camelot --pages 1,2,3 lattice your.pdf

The ``pages`` keyword argument accepts pages as comma-separated string of page numbers. You can also specify page ranges — for example, ``pages=1,4-10,20-30`` or ``pages=1,4-10,20-end``.

Reading encrypted PDFs
----------------------

To extract tables from encrypted PDF files you must provide a password when calling :meth:`read_pdf() <camelot.read_pdf>`.

::

    >>> tables = camelot.read_pdf('foo.pdf', password='userpass')
    >>> tables
    <TableList n=1>

.. tip::
    Here's how you can do the same with the :ref:`command-line interface <cli>`.
    ::

        $ camelot --password userpass lattice foo.pdf

Currently Camelot only supports PDFs encrypted with ASCII passwords and algorithm `code 1 or 2`_. An exception is thrown if the PDF cannot be read. This may be due to no password being provided, an incorrect password, or an unsupported encryption algorithm.

Further encryption support may be added in future, however in the meantime if your PDF files are using unsupported encryption algorithms you are advised to remove encryption before calling :meth:`read_pdf() <camelot.read_pdf>`. This can been successfully achieved with third-party tools such as `QPDF`_.

::

    $ qpdf --password=<PASSWORD> --decrypt input.pdf output.pdf

.. _code 1 or 2: https://github.com/mstamy2/PyPDF2/issues/378
.. _QPDF: https://www.github.com/qpdf/qpdf

----

Ready for more? Check out the :ref:`advanced <advanced>` section.
