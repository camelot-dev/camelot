.. _install:

Installation
============

This part of the documentation covers the steps to install Camelot.

.. note:: as of ``v1.0.0`` ghostscript is replaced by `pdfium <https://pypdfium2.readthedocs.io/en/stable/>`_ as the default image conversion backend. This should make the library easier to install with just a pip install (on linux). The other image conversion backends can still be used and are now optional to install.

You can use one of the following methods to install Camelot:

pip
---

To install Camelot from PyPI using ``pip``::

    $ pip install "camelot-py"

.. warning::

   Camelot depends on ``opencv-python-headless`` (the GUI-less OpenCV
   build). If your environment already has the full ``opencv-python``
   package installed, pip will let both sit side-by-side and the two
   shadow each other in ``site-packages``, which leads to broken
   ``import cv2`` at runtime. Uninstall the conflicting package first::

       $ pip uninstall opencv-python
       $ pip install camelot-py

   See `issue #645 <https://github.com/camelot-dev/camelot/issues/645>`_.

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
    $ pip install "."

Optional Dependencies
---------------------

Additional dependencies for Camelot can be installed using the following options

- ``[plot]`` installs the python package ``matplotlib`` and is used for :ref:`visual debugging <visual_debug>`.

-  ``[ghostscript]`` installs the python package ``ghostscript`` and is used for the optional ghostscript backend.

Note that ``[ghostscript]`` only installs the python package ``ghostscript``, which provides an interface to the Ghostscript C-API. Users must still `download <https://www.ghostscript.com/>`_ and install Ghostscript manually.
