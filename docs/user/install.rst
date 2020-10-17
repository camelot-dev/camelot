.. _install:

Installation of Camelot
=======================

This part of the documentation covers the steps to install Camelot.

After :ref:`installing the dependencies <install_deps>`, which include `Ghostscript <https://www.ghostscript.com>`_ and `Tkinter <https://wiki.python.org/moin/TkInter>`_, you can use one of the following methods to install Camelot:

.. warning:: The ``lattice`` flavor will fail to run if Ghostscript is not installed. You may run into errors as shown in `issue #193 <https://github.com/camelot-dev/camelot/issues/193>`_.

pip
---

To install Camelot from PyPI using ``pip``, please include the extra ``cv`` requirement as shown::

    $ pip install "camelot-py[cv]"

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
    $ pip install ".[cv]"
