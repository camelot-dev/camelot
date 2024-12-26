.. _install_deps:

Installation of dependencies
============================

.. note:: as of ``v1.0.0`` ghostscript is replaced by `pdfium <https://pypdfium2.readthedocs.io/en/stable/>`_ as the default image conversion backend. Which should make this library easier to install with just a pip install (on linux). The other imageconversion backends can still be used and are now optional to install.


The optional dependency `Ghostscript <https://www.ghostscript.com>`_ can be installed using your system's package manager or by running their installer.

OS-specific instructions
------------------------

Ubuntu
^^^^^^
.. code-block:: console

    $ apt install ghostscript

MacOS
^^^^^

.. code-block:: console

    $ brew install ghostscript

.. note::
  You might encounter the problem that the ghostscript module cannot be found. This can be fixed with the following commands.

  ``mkdir -p ~/lib``

  ``ln -s "$(brew --prefix gs)/lib/libgs.dylib" ~/lib``

Windows
^^^^^^^

For Ghostscript, you can get the installer at their `downloads page <https://www.ghostscript.com/download/gsdnld.html>`_. And for Tkinter, you can download the `ActiveTcl Community Edition <https://www.activestate.com/activetcl/downloads>`_ from ActiveState.

Checks to see if dependencies are installed correctly
-----------------------------------------------------

You can run the following checks to see if the dependencies were installed correctly.

For Ghostscript
^^^^^^^^^^^^^^^

Open the Python REPL and run the following:

For Ubuntu/MacOS

.. code-block:: pycon

    >>> from ctypes.util import find_library
    >>> find_library("gs")
    "libgs.so.9"

For Windows

.. code-block:: pycon

    >>> import ctypes
    >>> from ctypes.util import find_library
    >>> find_library("".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll")))
    <name-of-ghostscript-library-on-windows>

**Check:** The output of the ``find_library`` function should not be empty.

If the output is empty, then it's possible that the Ghostscript library is not available one of the ``LD_LIBRARY_PATH``/``DYLD_LIBRARY_PATH``/``PATH`` variables depending on your operating system. In this case, you may have to modify one of those path variables.
