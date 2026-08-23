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

- ``[ml]`` installs ``torch``, ``transformers`` and ``timm`` for the neural
  :ref:`flavor='ml' <ml>` (Table Transformer) backend, which recovers
  borderless tables.

- ``[ocr]`` installs ``rapidocr-onnxruntime``, the OCR text source that
  ``flavor='ml'`` uses on scanned / image-only PDFs. See
  :ref:`ocr_opencv_conflict` below before installing it.

Note that ``[ghostscript]`` only installs the python package ``ghostscript``, which provides an interface to the Ghostscript C-API. Users must still `download <https://www.ghostscript.com/>`_ and install Ghostscript manually.

.. _ocr_opencv_conflict:

The ``[ocr]`` extra and ``opencv-python``
-----------------------------------------

Camelot depends on ``opencv-python-headless``, but ``rapidocr-onnxruntime``
(pulled in by ``[ocr]``) hard-requires the full ``opencv-python``. So
``pip install "camelot-py[ocr]"`` installs **both** OpenCV distributions: they
ship the same ``cv2`` package and shadow each other in ``site-packages``, the
non-headless build wants GUI libraries (``libGL``) that slim containers do not
have, and it costs an extra ~70 MB download. This is the same conflict as the
warning above.

Nothing in the packaging metadata lets Camelot substitute one distribution for
another, and RapidOCR has declined to move to the headless build
(`RapidAI/RapidOCR#185 <https://github.com/RapidAI/RapidOCR/issues/185>`_), so
the override has to happen in *your* project. RapidOCR only ever does
``import cv2``, which ``opencv-python-headless`` satisfies, so dropping the
full build is safe.

With ``uv``, override the dependency away with a never-true marker in your
own ``pyproject.toml``::

    [tool.uv]
    override-dependencies = ["opencv-python ; sys_platform == 'never'"]

Then ``uv add "camelot-py[ocr]"`` resolves with ``opencv-python-headless``
only. The same override works ad hoc via an overrides file::

    $ echo 'opencv-python ; sys_platform == "never"' > overrides.txt
    $ uv pip install --override overrides.txt "camelot-py[ocr]"

``pdm`` users can exclude it instead::

    [tool.pdm.resolution]
    excludes = ["opencv-python"]

``pip`` has no override mechanism, so install RapidOCR without its
dependencies and supply them yourself::

    $ pip install "camelot-py"
    $ pip install --no-deps rapidocr-onnxruntime
    $ pip install pyclipper "numpy<3" six "shapely!=2.0.4" PyYAML Pillow \
        "onnxruntime>=1.7.0" tqdm

If you already installed both, reinstall the headless build cleanly — removing
one package deletes ``cv2`` files the other still needs::

    $ pip uninstall -y opencv-python opencv-python-headless
    $ pip install opencv-python-headless
