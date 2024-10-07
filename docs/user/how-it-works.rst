.. _how_it_works:

How It Works
============

This part of the documentation includes a high-level explanation of how pypdf_table_extraction extracts tables from PDF files.

You can choose between the following table parsing methods, *Stream*, *Lattice*, *Network* and *Hybrid*.
Where *Hybrid* is a combination of the *Network* and *Lattice* parser.

.. _stream:

Stream
------

Stream can be used to parse tables that have whitespaces between cells to simulate a table structure. It is built on top of PDFMiner's functionality of grouping characters on a page into words and sentences, using `margins <https://euske.github.io/pdfminer/#tools>`_.

1. Words on the PDF page are grouped into text rows based on their *y* axis overlaps.

2. Textedges are calculated and then used to guess interesting table areas on the PDF page. You can read `Anssi Nurminen's master's thesis <https://pdfs.semanticscholar.org/a9b1/67a86fb189bfcd366c3839f33f0404db9c10.pdf>`_ to know more about this table detection technique. [See pages 20, 35 and 40]

3. The number of columns inside each table area are then guessed. This is done by calculating the mode of number of words in each text row. Based on this mode, words in each text row are chosen to calculate a list of column *x* ranges.

4. Words that lie inside/outside the current column *x* ranges are then used to extend the current list of columns.

5. Finally, a table is formed using the text rows' *y* ranges and column *x* ranges and words found on the page are assigned to the table's cells based on their *x* and *y* coordinates.

.. _lattice:

Lattice
-------

Lattice is more deterministic in nature, and it does not rely on guesses. It can be used to parse tables that have demarcated lines between cells, and it can automatically parse multiple tables present on a page.

It starts by converting the PDF page to an image using ghostscript, and then processes it to get horizontal and vertical line segments by applying a set of morphological transformations (erosion and dilation) using OpenCV.

Let's see how Lattice processes the second page of `this PDF`_, step-by-step.

.. _this PDF: ../_static/pdf/us-030.pdf

1. Line segments are detected.

.. image:: ../_static/png/plot_line.png
    :height: 674
    :width: 1366
    :scale: 50%
    :align: center

2. Line intersections are detected, by overlapping the detected line segments and "`and`_"ing their pixel intensities.

.. _and: https://en.wikipedia.org/wiki/Logical_conjunction

.. image:: ../_static/png/plot_joint.png
    :height: 674
    :width: 1366
    :scale: 50%
    :align: center

3. Table boundaries are computed by overlapping the detected line segments again, this time by "`or`_"ing their pixel intensities.

.. _or: https://en.wikipedia.org/wiki/Logical_disjunction

.. image:: ../_static/png/plot_contour.png
    :height: 674
    :width: 1366
    :scale: 50%
    :align: center

4. Since dimensions of the PDF page and its image vary, the detected table boundaries, line intersections, and line segments are scaled and translated to the PDF page's coordinate space, and a representation of the table is created.

.. image:: ../_static/png/table.png
    :height: 674
    :width: 1366
    :scale: 50%
    :align: center

5. Spanning cells are detected using the line segments and line intersections.

.. image:: ../_static/png/plot_table.png
    :height: 674
    :width: 1366
    :scale: 50%
    :align: center

6. Finally, the words found on the page are assigned to the table's cells based on their *x* and *y* coordinates.

.. _network:

Network
------

The network parser is text-based: it relies on the bounding boxes of the text elements encoded in the .pdf document to identify patterns indicative of a table.

The plot belows shows the bounding boxes of all the text elements on the parsed document, in light blue for horizontal elements, light red for vertical elements (rare in most documents).

1. The network parser starts by identifying common horizontal or vertical coordinate alignments across these text elements. In other words it looks for bounding box rectangles which either share the same top, center, or bottom coordinates (horizontal axis), or the same left, right, or middle coordinates (vertical axis). See the generate method.

Once the parser found these alignments, it performs some pruning to only keep text elements that are part of a network - they have connections along both axis The idea is that it's not enough for two elements to be aligned to belong to a table, for instance the lines of text in this paragraph are all left-aligned, but they do not form a network. The pruning is done iteratively, see "remove_unconnected_edges" method.

Once the network is pruned, the parser keeps track of how many alignments each text element belongs to: that's the number on top (vertical alignments) or to the left of each alignment in the plot below. The text element with the most connections (in red on the plot) is the starting point -the seed- of the next step. Finally, the parser measures how far the alignments are from one another, to determine a plausible search zone around each cell for the next stage of growing the table. See "compute_plausible_gaps" method.

2. n the next step, the parser iteratively "grows" a table, starting from the seed identified in the previous step. The bounding box is initialized with the bounding box of the seed, then it iteratively searches for text elements that are close to the bounding box, then grows the table to ingest them, until there are no more text elements to ingest. The two steps are:

Search: create a search bounding box by expanding the current table bounding box in all directions, based on the plausible gap numbers determined above.
Grow: if a networked text element is found in this search area, expand the table bounding box so that it includes this new element.

The search area and the table bounding box grow starting from the seed. See method "search_table_body".

3. Headers are often aligned differently from the rest of the table. To account for this, the network parser searches for text elements that are good candidates for a header section: these text elements are just above the bounding box of the body of the table, and they fit within the rows identified in the table body. See the method "search_header_from_body_bbox".

4. Words that lie inside/outside the current column *x* ranges are then used to extend the current list of columns.

5. There are sometimes multiple tables on one page. So once a first table is identified, all the text edges it contains are removed, and the algorithm is repeated until no new network is identified.

.. _hybrid:

Hybrid
------

The hybrid parser  aims to combine the strengths of the Network parser (identifying cells based on text alignments) and of the Lattice parser (relying on solid lines to determine tables rows and columns boundaries).

1. Hybrid calls both parsers, to get a) the standard table parse, b) the coordinates of the rows and columns boundaries, and c) the table boundaries (or contour).

2. If there are areas in the document where both lattice and network found a table, the hybrid parser uses the results from network, but enhances them based on the rows/columns boundaries identified by lattice in the area. Because lattice uses the solid lines detected on the document, the coordinates for b) and c) detected by Lattice are generally more precise. See the "_merge_bbox_analysis" method.
