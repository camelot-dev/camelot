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
