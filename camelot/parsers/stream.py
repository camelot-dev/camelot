# -*- coding: utf-8 -*-

from __future__ import division

from .base import TextBaseParser
from ..core import TextEdges
from ..utils import (
    bbox_from_str,
    text_in_bbox
)


class Stream(TextBaseParser):
    """Stream method of parsing looks for spaces between text
    to parse the table.

    If you want to specify columns when specifying multiple table
    areas, make sure that the length of both lists are equal.

    Parameters
    ----------
    table_regions : list, optional (default: None)
        List of page regions that may contain tables of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    table_areas : list, optional (default: None)
        List of table area strings of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    columns : list, optional (default: None)
        List of column x-coordinates strings where the coordinates
        are comma-separated.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    edge_tol : int, optional (default: 50)
        Tolerance parameter for extending textedges vertically.
    row_tol : int, optional (default: 2)
        Tolerance parameter used to combine text vertically,
        to generate rows.
    column_tol : int, optional (default: 0)
        Tolerance parameter used to combine text horizontally,
        to generate columns.

    """

    def __init__(
            self,
            table_regions=None,
            table_areas=None,
            columns=None,
            flag_size=False,
            split_text=False,
            strip_text="",
            edge_tol=50,
            row_tol=2,
            column_tol=0,
            **kwargs):
        super().__init__(
            "stream",
            table_regions=table_regions,
            table_areas=table_areas,
            columns=columns,
            flag_size=flag_size,
            split_text=split_text,
            strip_text=strip_text,
            edge_tol=edge_tol,
            row_tol=row_tol,
            column_tol=column_tol,
        )
        self.textedges = []

    def _nurminen_table_detection(self, textlines):
        """A general implementation of the table detection algorithm
        described by Anssi Nurminen's master's thesis.
        Link: https://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3 # noqa

        Assumes that tables are situated relatively far apart
        vertically.
        """
        # TODO: add support for arabic text #141
        # sort textlines in reading order
        textlines.sort(key=lambda x: (-x.y0, x.x0))
        textedges = TextEdges(edge_tol=self.edge_tol)
        # generate left, middle and right textedges
        textedges.generate(textlines)
        # select relevant edges
        relevant_textedges = textedges.get_relevant()
        self.textedges.extend(relevant_textedges)
        # guess table areas using textlines and relevant edges
        table_bbox = textedges.get_table_areas(textlines, relevant_textedges)
        # treat whole page as table area if no table areas found
        if not table_bbox:
            table_bbox = {(0, 0, self.pdf_width, self.pdf_height): None}

        return table_bbox

    def record_parse_metadata(self, table):
        """Record data about the origin of the table
        """
        super().record_parse_metadata(table)
        table._textedges = self.textedges

    def _generate_table_bbox(self):
        if self.table_areas is None:
            hor_text = self.horizontal_text
            if self.table_regions is not None:
                # filter horizontal text
                hor_text = []
                for region_str in self.table_regions:
                    region_text = text_in_bbox(
                        bbox_from_str(region_str),
                        self.horizontal_text)
                    hor_text.extend(region_text)
            # find tables based on nurminen's detection algorithm
            table_bbox = self._nurminen_table_detection(hor_text)
        else:
            table_bbox = {}
            for area_str in self.table_areas:
                table_bbox[bbox_from_str(area_str)] = None
        self.table_bbox = table_bbox
