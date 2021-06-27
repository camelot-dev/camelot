.. _install_deps:

Installation of dependencies
============================

The dependencies `Ghostscript <https://www.ghostscript.com>`_ and `Tkinter <https://wiki.python.org/moin/TkInter>`_ can be installed using your system's package manager or by running their installer.

OS-specific instructions
------------------------

Ubuntu
^^^^^^
::

    $ apt install ghostscript python3-tk

MacOS
^^^^^
::

    $ brew install ghostscript tcl-tk

Windows
^^^^^^^

For Ghostscript, you can get the installer at their `downloads page <https://www.ghostscript.com/download/gsdnld.html>`_. And for Tkinter, you can download the `ActiveTcl Community Edition <https://www.activestate.com/activetcl/downloads>`_ from ActiveState.

Checks to see if dependencies are installed correctly
-----------------------------------------------------

You can run the following checks to see if the dependencies were installed correctly.

For Ghostscript
^^^^^^^^^^^^^^^

Open the Python REPL and run the following:

For Ubuntu/MacOS::

    >>> from ctypes.util import find_library
    >>> find_library("gs")
    "libgs.so.9"

For Windows::

    >>> import ctypes
    >>> from ctypes.util import find_library
    >>> find_library("".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll")))
    <name-of-ghostscript-library-on-windows>

**Check:** The output of the ``find_library`` function should not be empty.

If the output is empty, then it's possible that the Ghostscript library is not available one of the ``LD_LIBRARY_PATH``/``DYLD_LIBRARY_PATH``/``PATH`` variables depending on your operating system. In this case, you may have to modify one of those path variables.

For Tkinter
^^^^^^^^^^^

Launch Python and then import Tkinter::

    >>> import tkinter

**Check:** Importing ``tkinter`` should not raise an import error.
