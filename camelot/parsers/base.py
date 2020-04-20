# -*- coding: utf-8 -*-

import os
import warnings

from ..utils import (
    get_text_objects,
    get_table_index,
    text_in_bbox,
    bbox_from_str,
)
from ..core import Table


class BaseParser(object):
    """Defines a base parser.
    """
    def __init__(
        self,
        parser_id,
        table_regions=None,
        table_areas=None,
        copy_text=None,
        split_text=False,
        strip_text="",
        shift_text=None,
        flag_size=False,
        debug=False
    ):
        self.id = parser_id
        self.table_regions = table_regions
        self.table_areas = table_areas

        self.copy_text = copy_text
        self.split_text = split_text
        self.strip_text = strip_text
        self.shift_text = shift_text

        self.flag_size = flag_size

        self.rootname = None
        self.t_bbox = None

        # For plotting details of parsing algorithms
        self.debug_info = {} if debug else None

    def prepare_page_parse(self, filename, layout, dimensions,
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

        if self.debug_info is not None:
            self.debug_info["table_regions"] = self.table_regions
            self.debug_info["table_areas"] = self.table_areas

    def _apply_regions_filter(self, textlines):
        """If regions have been specified, filter textlines to these regions.

        Parameters
        ----------
        textlines : list
            list of textlines to be filtered

        Returns
        -------
        filtered_textlines : list of textlines within the regions specified

        """
        filtered_textlines = []
        if self.table_regions is None:
            filtered_textlines.extend(textlines)
        else:
            for region_str in self.table_regions:
                region_text = text_in_bbox(
                    bbox_from_str(region_str),
                    textlines
                )
                filtered_textlines.extend(region_text)
        return filtered_textlines

    def _document_has_no_text(self):
        """Detects image only documents and warns.

        Returns
        -------
        has_no_text : bool
            Whether the document doesn't have any text at all.
        """
        if not self.horizontal_text:
            rootname = os.path.basename(self.rootname)
            if self.images:
                warnings.warn(
                    "{rootname} is image-based, "
                    "camelot only works on text-based pages."
                    .format(rootname=rootname)
                )
            else:
                warnings.warn(
                    "No tables found on {rootname}".format(rootname=rootname)
                )
            return True
        return False

    def _initialize_new_table(self, table_idx, cols, rows):
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

    def compute_parse_errors(self, table):
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
