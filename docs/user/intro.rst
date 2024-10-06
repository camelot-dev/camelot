.. _intro:

Introduction
============

The Camelot Project
-------------------

The PDF (Portable Document Format) was born out of `The Camelot Project`_ to create "a universal way to communicate documents across a wide variety of machine configurations, operating systems and communication networks". The goal was to make these documents viewable on any display and printable on any modern printers. The invention of the `PostScript`_ page description language, which enabled the creation of *fixed-layout* flat documents (with text, fonts, graphics, images encapsulated), solved this problem.

At a high level, PostScript defines instructions, such as "place this character at this *x,y* coordinate on a plane". Spaces can be *simulated* by placing characters relatively far apart. Extending from that, tables can be *simulated* by placing characters (which constitute words) in two-dimensional grids. A PDF viewer just takes these instructions and draws everything for the user to view. Since a PDF is just characters on a plane, there is no table data structure that can be extracted and used for analysis!

Sadly, a lot of today's open data is trapped in PDF tables.

.. _PostScript: https://web.archive.org/web/20210203041543/http://www.planetpdf.com/planetpdf/pdfs/warnock_camelot.pdf

Why another PDF table extraction library?
-----------------------------------------

There are both open (`Tabula`_, `pdf-table-extract`_) and closed-source (`smallpdf`_, `PDFTables`_) tools that are widely used to extract tables from PDF files. They either give a nice output or fail miserably. There is no in between. This is not helpful since everything in the real world, including PDF table extraction, is fuzzy. This leads to the creation of ad-hoc table extraction scripts for each type of PDF table.

pypdf_table_extraction (formerly `Camelot`_) was created to offer users complete control over table extraction. If you can't get your desired output with the default settings, you can tweak them and get the job done!

Here is a `comparison`_ of pypdf_table_extraction's output with outputs from other open-source PDF parsing libraries and tools.

.. _Tabula: http://tabula.technology/
.. _pdf-table-extract: https://github.com/ashima/pdf-table-extract
.. _PDFTables: https://pdftables.com/
.. _Smallpdf: https://smallpdf.com
.. _comparison: https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools

What's in a name?
-----------------

pypdf_table_extraction is the comunity maintained fork of `Camelot`_.
As you can already guess, that library is named after `The Camelot Project`_.

.. note::
  Fun fact: In the British comedy film `Monty Python and the Holy Grail`_ (and in the `Arthurian legend`_ depicted in the film), "Camelot" is the name of the castle where Arthur leads his men, the Knights of the Round Table, and then sets off elsewhere after deciding that it is "a silly place". Interestingly, the language in which this library is written (Python) was named after Monty Python.

.. _Camelot: https://github.com/camelot-dev/camelot
.. _The Camelot Project: https://web.archive.org/web/20210203041543/http://www.planetpdf.com/planetpdf/pdfs/warnock_camelot.pdf
.. _Monty Python and the Holy Grail: https://en.wikipedia.org/wiki/Monty_Python_and_the_Holy_Grail
.. _Arthurian legend: https://en.wikipedia.org/wiki/King_Arthur

pypdf_table_extracion License
-----------------------------

    .. include:: ../../LICENSE
