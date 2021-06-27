.. _faq:

Frequently Asked Questions
==========================

This part of the documentation answers some common questions. To add questions, please open an issue `here <https://github.com/camelot-dev/camelot/issues/new>`_.

How to reduce memory usage for long PDFs?
-----------------------------------------

During table extraction from long PDF documents, RAM usage can grow significantly.

A simple workaround is to divide the extraction into chunks, and save extracted data to disk at the end of every chunk.

For more details, check out this code snippet from `@anakin87 <https://github.com/anakin87>`_:

::

    import camelot


    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i : i + n]


    def extract_tables(filepath, pages, chunks=50, export_path=".", params={}):
        """
        Divide the extraction work into n chunks. At the end of every chunk,
        save data on disk and free RAM.

        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.
        """

        # get list of pages from camelot.handlers.PDFHandler
        handler = camelot.handlers.PDFHandler(filepath)
        page_list = handler._get_pages(filepath, pages=pages)

        # chunk pages list
        page_chunks = list(chunks(page_list, chunks))

        # extraction and export
        for chunk in page_chunks:
            pages_string = str(chunk).replace("[", "").replace("]", "")
            tables = camelot.read_pdf(filepath, pages=pages_string, **params)
            tables.export(f"{export_path}/tables.csv")
