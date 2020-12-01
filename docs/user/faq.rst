.. _faq:

FAQ
===

This part of the documentation answers some common questions. If you want to add some questions you can simply open as issue `here <https://github.com/camelot-dev/camelot/issues/new>`_.


How could you Optimize memory usage for long PDFs ?
---------------------------------------------------


In order to optimize memory usage you need to flush tables every ``n`` pages. For more information refer this snippet of code from `@anakin87 <https://github.com/anakin87>`_.

.. code-block:: python3


        # These long PDF contain regional balance sheets.
        # Code (chunked extraction) is similar to this:

        from main import logger
        import camelot
        import shutil
        import pathlib
        import json
        import os
        import glob

        def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
        yield l[i:i + n]

        def extract_tables_from_path(filename,pages,params=None):

        std_params_set = {
            'flavor': 'lattice',
            'line_scale': 65,
            'copy_text': ['h', 'v'],
            'split_text': True
        }

        # keys to export in JSON
        selected_keys = ['rows',
                        'whitespace',
                        '_bbox',
                        'cols',
                        'page',
                        'shape',
                        'flavor',
                        'order',
                        'accuracy']

        logger.info('\n\n' + '*' * 50 + 'START' + '*' * 50)
        logger.info('WORKING ON FILE {}'.format(filename))


        dir_name = filename.rpartition('/')[-1].rpartition('.')[0]
        dir_path = '/comuni-data/exp_tables/' + dir_name
        dir_temp=dir_path+'/temp'

        # Clean and recreate output directory
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            pathlib.Path(dir_temp).mkdir(parents=True, exist_ok=True)
        except:
            logger.exception('Error in cleaning/creating output directory')
            return None  

        params_set = params if params else std_params_set

        logger.info('USING THE FOLLOWING PARAMETERS: {}'.format(params_set))        


        # Control page number (by Camelot method)
        # and subdivide the extraction work into 50-pages parts.
        # AT THE END OF EVERY PART, SAVE THE DATA ON DISk AND FREE RAM

        handler=camelot.handlers.PDFHandler(filename)
        pages_list=handler._get_pages(filename,pages=pages)

        pages_chunks=list(chunks(pages_list,50))

        last_index=0
        tot_tables=0
        index=0



        for chunk in pages_chunks:
            tables=[]
            new_table_list=[]

            pages_string=str(chunk).replace('[','').replace(']','')



            try:
                tables = camelot.read_pdf(filename, pages=pages_string, **params_set)
            except Exception as e:
                logger.exception('ERROR IN TABLE EXTRACTION')
                return None




            # First filter      
            new_table_list =[table for table in tables if table.shape != (1, 1)]

            # Second filter

            new_table_list = [table for table in new_table_list if (table.parsing_report['accuracy'] > 75 \
                            or table.parsing_report['accuracy'] < 0) \
                            and table.parsing_report['whitespace'] < 80\
                            and '(cid:' not in str(table.data)]



            logger.info('SAVING EXTRACTION')

            # Exports in JSON the selected fields
            tables_bboxes = []

            for index, table in enumerate(new_table_list,last_index):
                table_dict = {key: table.__dict__[key] for key in selected_keys}

                table_dict['id'] = index
                table_dict['filepath'] = filename
                table_dict['json_data'] = table.__dict__['df'].to_json()

                table_filename = '{}/table-{}.json'.format(dir_path, index)


                with open(table_filename, "w") as file:
                    json.dump(table_dict, file)

                
            last_index=index
            tot_tables+=len(new_table_list)


            

        logger.info('{} VALID TABLES DETECTED'.format(tot_tables))
        logger.info('*' * 50 + 'END' + '*' * 50)        

        api_response=ApiResponse(n_of_valid_tables=tot_tables,output_directory=str(pathlib.Path(dir_path).resolve()))


        return api_response
