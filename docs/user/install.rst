.. _install:

Installation
============

This part of the documentation covers the steps to install Camelot.

.. note:: as of ``v1.0.0`` ghostscript is replaced by `pdfium <https://pypdfium2.readthedocs.io/en/stable/>`_ as the default image conversion backend. This should make the library easier to install with just a pip install (on linux). The other image conversion backends can still be used and are now optional to install.

You can use one of the following methods to install Camelot:

pip
---

To install Camelot from PyPI using ``pip``, please include the extra ``cv`` requirement as shown::

    $ pip install "camelot-py[base]"

conda
-----

`conda`_ is a package manager and environment management system for the `Anaconda <https://anaconda.org>`_ distribution. It can be used to install Camelot from the ``conda-forge`` channel::

    $ conda install -c conda-forge camelot-py

From the source code
--------------------

After :ref:`installing the dependencies <install_deps>`, you can install Camelot from source by:

1. Cloning the GitHub repository.
::

    $ git clone https://www.github.com/camelot-dev/camelot

2. And then simply using pip again.
::

    $ cd camelot
    $ pip install ".[base]"
