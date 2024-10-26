.. _install:

Installation
=============

This part of the documentation covers the steps to install pypdf_table_extraction.

.. note:: ``ghostscript`` is replaced by ``pdfium`` as the default image conversion backend in ``v1.0.0``. Which should make this library easier to install with just a pip install (on linux).

You can use one of the following methods to install pypdf_table_extraction:


pip
---

To install pypdf_table_extraction from PyPI using ``pip``

.. code-block:: console

    $ pip install "pypdf-table-extraction[base]"

conda
-----


`conda`_ is a package manager and environment management system for the `Anaconda <https://anaconda.org>`_ distribution. It can be used to install pypdf_table_extraction from the ``conda-forge`` channel

.. code-block:: console

    $ conda install -c conda-forge pypdf-table-extraction

From the source code
--------------------

You can install pypdf_table_extraction from source by:

1. Cloning the GitHub repository.

.. code-block:: console

    $ git clone https://github.com/py-pdf/pypdf_table_extraction.git

2. And then simply using pip again.

.. code-block:: console

    $ cd camelot
    $ pip install ".[base]"

.. tip::
    You can still use the `ghostscript`` backend after After :ref:`installing the dependencies <install_deps>`.
