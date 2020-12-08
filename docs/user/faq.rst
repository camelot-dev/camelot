.. _faq:

FAQ
===

This part of the documentation answers some common questions. If you want to add some questions you can simply open an issue `here <https://github.com/camelot-dev/camelot/issues/new>`_.


How to reduce memory usage for long PDFs?
---------------------------------------------------

During table extraction from long PDF documents, RAM usage can grow significantly.
 
A simple workaround is to divide the extraction into some chunks (for example, chunks of 50 pages); at the end of every chunk extraction, data are saved to disk.

For more information, refer to this code snippet from `@anakin87 <https://github.com/anakin87>`_:

.. code-block:: python3

    import camelot
    
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    		
    def extract_tables_with_less_memory_usage(filepath, pages, params={}, 
 export_path='.', chunks_length=50):
        """
        Control page number
        and subdivide the extraction work into n-pages parts (chunks_length).
        At the end of every part, save the data on disk and free ram
        """
    
        # get list of document pages from Camelot handler
        handler=camelot.handlers.PDFHandler(filepath)
        pages_list=handler._get_pages(filepath,pages=pages)
        
        # chunk pages list
        pages_chunks=list(chunks(pages_list,chunks_length))
    
        # extraction and export
        for chunk in pages_chunks:
            pages_string=str(chunk).replace('[','').replace(']','')
            tables = camelot.read_pdf(filepath, pages=pages_string,**params)
            tables.export(f'{export_path}/tables.json',f='json')	
