.. _faq:

Frequently Asked Questions
==========================

This part of the documentation answers some common questions. To add questions, please open an issue `here <https://github.com/camelot-dev/camelot/issues/new>`_.

Does Camelot work with image-based PDFs?
----------------------------------------

**No**, Camelot only works with text-based PDFs and not scanned documents. (As Tabula `explains <https://github.com/tabulapdf/tabula#why-tabula>`_, "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

How to reduce memory usage for long PDFs?
-----------------------------------------

When extracting tables from a long PDF in one call, RAM grows roughly
with the number of pages held in memory at once: every page's text
objects, the per-parser caches, and the resulting :class:`TableList` all
live until ``read_pdf`` returns.

The simplest mitigation is to process the document in page-range
chunks, write each chunk's tables to disk, and let Python free the
intermediate state between calls. The pattern below uses only the
public API and works on any range of pages — concept originally from
`#90 <https://github.com/camelot-dev/camelot/pull/90>`_ by
`@nightwarriorftw <https://github.com/nightwarriorftw>`_ and
`@anakin87 <https://github.com/anakin87>`_::

    import camelot


    def extract_in_chunks(
        filepath,
        total_pages,
        chunk_size=50,
        export_dir=".",
        **read_pdf_kwargs,
    ):
        """Extract tables a chunk of pages at a time, freeing RAM between chunks.

        Parameters
        ----------
        filepath : str
            Path to the PDF file.
        total_pages : int
            Total page count of the PDF. Get it from any PDF tool, e.g.
            ``len(playa.parse(open(filepath, "rb").read()).pages)``.
        chunk_size : int, optional (default: 50)
            How many pages to process per ``read_pdf`` call.
        export_dir : str, optional (default: ".")
            Directory in which per-chunk CSVs are written.
        **read_pdf_kwargs
            Any other keyword arguments are forwarded to
            :meth:`camelot.read_pdf` (e.g. ``flavor="stream"``,
            ``table_areas=...``).
        """
        for start in range(1, total_pages + 1, chunk_size):
            end = min(start + chunk_size - 1, total_pages)
            tables = camelot.read_pdf(
                filepath, pages=f"{start}-{end}", **read_pdf_kwargs
            )
            tables.export(f"{export_dir}/tables_{start}-{end}.csv")

Each iteration's :class:`TableList` becomes unreachable after the
``tables.export(...)`` call, so the per-chunk PDF parse state is
released before the next chunk runs. For very long documents, combine
this with ``flavor="stream"`` or ``flavor="network"`` (cheaper than
Lattice's image conversion) where the table layouts allow it.

How can I supply my own image conversion backend to Lattice?
------------------------------------------------------------

When using the :ref:`Lattice <lattice>` flavor, you can supply your own :ref:`image conversion backend <image-conversion-backend>` by creating a class with a ``convert`` method as follows::

    >>> class ConversionBackend(object):
    >>>     def convert(pdf_path, png_path):
    >>>         # read pdf page from pdf_path
    >>>         # convert pdf page to image
    >>>         # write image to png_path
    >>>         pass
    >>>
    >>> tables = camelot.read_pdf(filename, backend=ConversionBackend())

Why don't a table's bbox coordinates line up with the page image?
-----------------------------------------------------------------

A ``Table``'s ``_bbox`` (and the coordinates in its cells) live in **PDF
coordinate space**, while ``table.get_pdf_image()`` returns the page as a
rendered raster in **image coordinate space**. The two differ in two ways,
so drawing ``_bbox`` straight onto the image puts the box in the wrong
place:

- **Origin / direction.** PDF space has its origin at the *bottom-left*
  with y increasing *upward*; image space has its origin at the
  *top-left* with y increasing *downward*. The y-axis must be flipped.
- **Scale.** PDF coordinates are in points (1/72 inch). The image is
  rendered at a higher resolution (300 dpi by default), so it is roughly
  ``300/72`` times larger on each axis.

``Table`` exposes the page size as ``table.pdf_size`` ``(width, height)``,
which together with the rendered image's pixel size gives the per-axis
scale. To overlay a table's bbox on its page image::

    import camelot
    import cv2

    tables = camelot.read_pdf("foo.pdf", flavor="lattice")
    table = tables[0]

    img = table.get_pdf_image()          # rendered raster (BGR)
    image_h, image_w = img.shape[:2]
    pdf_w, pdf_h = table.pdf_size
    scale_x, scale_y = image_w / pdf_w, image_h / pdf_h

    x0, y0, x1, y1 = table._bbox         # PDF coords (origin bottom-left)
    top_left = (round(x0 * scale_x), round((pdf_h - y1) * scale_y))
    bottom_right = (round(x1 * scale_x), round((pdf_h - y0) * scale_y))

    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.imwrite("foo_bbox.jpg", img)

The same scale-and-flip converts any PDF-space coordinate (a cell, a
joint) into the image, and inverting it converts an image-space
coordinate back into PDF space.

I have table coordinates from an image — how do I pass them to ``table_areas``?
------------------------------------------------------------------------------

A common workflow is to detect a table's region with an external,
image-based tool (an ML layout detector such as table-transformers, or
manual annotation on a rendered page) and then have Camelot extract just
that region via ``table_areas``. Because ``table_areas`` is in **PDF
coordinate space** (origin bottom-left, points), you have to convert your
**image** coordinates the other way — the inverse of the section above.

The key detail: use the DPI of *your own* render. If you rasterised the
page yourself (e.g. with ``pdf2image`` at ``dpi=D``), then
``image_width = pdf_width * D / 72``, so the per-axis scale back to PDF
points is ``72 / D`` — **not** Camelot's internal 300 dpi.

::

    import camelot

    # Box from an image-based detector: (left, top, right, bottom) in
    # pixels, origin top-left, on a page you rendered at `dpi`.
    x0_img, y0_img, x1_img, y1_img = detected_box
    dpi = 200  # whatever you passed to pdf2image / your renderer

    # image px -> PDF points
    s = 72.0 / dpi
    pdf_w_x0, pdf_w_x1 = x0_img * s, x1_img * s
    # flip y: image top-left origin -> PDF bottom-left origin.
    # page height in points = image_height_px * 72 / dpi
    page_h_pts = image_height_px * s
    pdf_top = page_h_pts - y0_img * s        # image top  -> larger PDF y
    pdf_bottom = page_h_pts - y1_img * s     # image bot  -> smaller PDF y

    # table_areas wants "x1,y1,x2,y2" = top-left, bottom-right (PDF space)
    area = f"{pdf_w_x0},{pdf_top},{pdf_w_x1},{pdf_bottom}"
    tables = camelot.read_pdf("doc.pdf", flavor="lattice", table_areas=[area])

If your detector ran on an image Camelot itself produced (rather than your
own ``pdf2image`` render), use ``table.pdf_size`` and the rendered image
size to get the scale, exactly as in the previous answer but inverted.

I get ``AttributeError: module 'camelot' has no attribute 'read_pdf'``
----------------------------------------------------------------------

This almost always means the **wrong package** is installed. There is an
unrelated project on PyPI also called ``camelot`` (a configuration
library); ``pip install camelot`` installs *that*, not this table-
extraction library. Uninstall it and install ``camelot-py``::

    $ pip uninstall camelot camelot-py
    $ pip install "camelot-py[base]"

The import name is still ``camelot`` (``import camelot``) — only the
*install* name differs (``camelot-py``).

If the right package is installed and you still hit this, check that you
don't have a local file named ``camelot.py`` (or a ``camelot/`` folder)
in your working directory shadowing the installed package — rename it and
try again.
