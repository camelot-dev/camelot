# -*- coding: utf-8 -*-

import os

from ..utils import (
    get_text_objects,
    get_table_index
)
from ..core import Table


class BaseParser(object):
    """Defines a base parser.
    """
    def __init__(self,
        parser_id,
        table_regions=None,
        table_areas=None,
        split_text=False,
        strip_text="",
        shift_text=None,
        flag_size=False,
    ):
        self.id = parser_id
        self.table_regions = table_regions
        self.table_areas = table_areas

        self.split_text = split_text
        self.strip_text = strip_text
        self.shift_text = shift_text

        self.flag_size = flag_size

        self.t_bbox = None

        # For plotting details of parsing algorithms
        self.debug_info = {}

    def _generate_layout(self, filename, layout, dimensions,
                         page_idx, layout_kwargs):
        self.filename = filename
        self.layout_kwargs = layout_kwargs
        self.layout = layout
        self.dimensions = dimensions
        self.page = page_idx
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
    table : camelot.core.Table

    """
    def _initialize_new_table(self, table_idx, cols, rows):
        table = Table(cols, rows)
        table.page = self.page
        table.order = table_idx + 1
        return table


    @staticmethod
    def _reduce_index(t, idx, shift_text):
        """Reduces index of a text object if it lies within a spanning
        cell.  Only useful for some parsers (e.g. Lattice), base method is a
        noop.
        """
        return idx

    def _compute_parse_errors(self, table):
        pos_errors = []
        # TODO: have a single list in place of two directional ones?
        # sorted on x-coordinate based on reading order i.e. LTR or RTL
        for direction in ["vertical", "horizontal"]:
            for t in self.t_bbox[direction]:
                indices, error = get_table_index(
                    table,
                    t,
                    direction,
                    split_text=self.split_text,
                    flag_size=self.flag_size,
                    strip_text=self.strip_text,
                )
                if indices[:2] != (-1, -1):
                    pos_errors.append(error)
                    indices = type(self)._reduce_index(
                        table,
                        indices,
                        shift_text=self.shift_text
                    )
                    for r_idx, c_idx, text in indices:
                        table.cells[r_idx][c_idx].text = text
        return pos_errors

