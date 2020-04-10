# -*- coding: utf-8 -*-

import os

from ..utils import (
    get_page_layout,
    get_text_objects
)
from ..core import Table

from ..image_processing import (
    adaptive_threshold,
    find_lines,
    find_contours,
    find_joints
)

# Pylint can't detect contents of cv2
from cv2 import imread  # pylint: disable=no-name-in-module


class BaseParser(object):
    """Defines a base parser.
    """
    def __init__(self, parser_id):
        self.imagename = None
        self.pdf_image = None
        self.id = parser_id

        # For plotting details of parsing algorithms
        self.debug_info = {}

    def _generate_layout(self, filename, page_idx, layout_kwargs):
        self.filename = filename
        self.layout_kwargs = layout_kwargs
        self.layout, self.dimensions = get_page_layout(
            filename,
            **layout_kwargs
        )
        self.images = get_text_objects(self.layout, ltype="image")
        self.horizontal_text = get_text_objects(
            self.layout,
            ltype="horizontal_text"
        )
        self.vertical_text = get_text_objects(
            self.layout,
            ltype="vertical_text"
        )
        self.pdf_width, self.pdf_height = self.dimensions
        self.rootname, __ = os.path.splitext(self.filename)

        self.page = page_idx

    def generate_image(self):
        if self.pdf_image is None:
            self._generate_image_file()
            self.pdf_image = imread(self.imagename)

    def _generate_image_file(self):
        if self.imagename:
            return
        from ..ext.ghostscript import Ghostscript

        self.imagename = "".join([self.rootname, ".png"])
        gs_call = "-q -sDEVICE=png16m -o {} -r300 {}".format(
            self.imagename, self.filename
        )
        gs_call = gs_call.encode().split()
        null = open(os.devnull, "wb")
        Ghostscript(*gs_call, stdout=null)
        # with Ghostscript(*gs_call, stdout=null) as gs:
        #     pass
        null.close()

    """Initialize new table object, ready to be populated

    Parameters
    ----------
    table_idx : int
        Index of this table within the pdf page analyzed
    cols : list
        list of coordinate boundaries tuples (left, right)
    rows : list
        list of coordinate boundaries tuples (bottom, top)

    Returns
    -------
    t : camelot.core.Table

    """
    def _initialize_new_table(self, table_idx, cols, rows):
        table = Table(cols, rows)
        table.page = self.page
        table.order = table_idx + 1
        return table
