.. _install:

Installation of Camelot
=======================

This part of the documentation covers the steps to install Camelot.

Using conda
-----------

The easiest way to install Camelot is to install it with `conda`_, which is a package manager and environment management system for the `Anaconda`_ distribution.
::

    $ conda install -c conda-forge camelot-py

.. note:: Camelot is available for Python 2.7, 3.5, 3.6 and 3.7 on Linux, macOS and Windows. For Windows, you will need to install ghostscript which you can get from their `downloads page`_.

.. _conda: https://conda.io/docs/
.. _Anaconda: http://docs.continuum.io/anaconda/
.. _downloads page: https://www.ghostscript.com/download/gsdnld.html
.. _conda-forge: https://conda-forge.org/

Using pip
---------

After :ref:`installing the dependencies <install_deps>`, which include `Tkinter`_ and `ghostscript`_, you can simply use pip to install Camelot::

    $ pip install camelot-py[cv]

.. _Tkinter: https://wiki.python.org/moin/TkInter
.. _ghostscript: https://www.ghostscript.com

From the source code
--------------------

After :ref:`installing the dependencies <install_deps>`, you can install from the source by:

1. Cloning the GitHub repository.
::

    $ git clone https://www.github.com/camelot-dev/camelot

2. Then simply using pip again.
::

    $ cd camelot
    $ pip install ".[cv]"
