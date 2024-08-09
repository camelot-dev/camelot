.. _install:

Installation of pypdf_table_extraction Camelot:
===============================================

This part of the documentation covers the steps to install pypdf_table_extraction.

After :ref:`installing the dependencies <install_deps>`, which include `Ghostscript <https://www.ghostscript.com>`_ and `Tkinter <https://wiki.python.org/moin/TkInter>`_, you can use one of the following methods to install pypdf_table_extraction:

.. warning:: The ``lattice`` flavor will fail to run if Ghostscript is not installed. You may run into errors as shown in `issue #193 <https://github.com/camelot-dev/camelot/issues/193>`_.

pip
---

To install pypdf_table_extraction from PyPI using ``pip``, please include the extra ``cv`` requirement as shown::

    $ pip install "pypdf-table-extraction[base]"

conda
-----

`conda`_ is a package manager and environment management system for the `Anaconda <https://anaconda.org>`_ distribution. It can be used to install pypdf_table_extraction from the ``conda-forge`` channel::

    $ conda install -c conda-forge pypdf-table-extraction

From the source code
--------------------

After :ref:`installing the dependencies <install_deps>`, you can install pypdf_table_extraction from source by:

1. Cloning the GitHub repository.
::

    $ git clone https://github.com/py-pdf/pypdf_table_extraction.git

2. And then simply using pip again.
::

    $ cd camelot
    $ pip install ".[base]"
