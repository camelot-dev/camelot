

# -*- coding: utf-8 -*-

from __future__ import division
import warnings

import numpy as np

from .base import BaseParser
from ..core import TextEdges
from ..utils import (text_in_bbox, text_in_bbox_per_axis)


class Hybrid(BaseParser):
    """Hybrid method of parsing looks for spaces between text
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
        **kwargs
    ):
        super().__init__(
            "hybrid",
            table_regions=table_regions,
            table_areas=table_areas,
            split_text=split_text,
            strip_text=strip_text,
            flag_size=flag_size,
        )
        self.columns = columns
        self._validate_columns()
        self.edge_tol = edge_tol
        self.row_tol = row_tol
        self.column_tol = column_tol

    @staticmethod
    def _text_bbox(t_bbox):
        """Returns bounding box for the text present on a page.

        Parameters
        ----------
        t_bbox : dict
            Dict with two keys 'horizontal' and 'vertical' with lists of
            LTTextLineHorizontals and LTTextLineVerticals respectively.

        Returns
        -------
        text_bbox : tuple
            Tuple (x0, y0, x1, y1) in pdf coordinate space.

        """
        xmin = min(t.x0 for direction in t_bbox for t in t_bbox[direction])
        ymin = min(t.y0 for direction in t_bbox for t in t_bbox[direction])
        xmax = max(t.x1 for direction in t_bbox for t in t_bbox[direction])
        ymax = max(t.y1 for direction in t_bbox for t in t_bbox[direction])
        text_bbox = (xmin, ymin, xmax, ymax)
        return text_bbox

    @staticmethod
    def _group_rows(text, row_tol=2):
        """Groups PDFMiner text objects into rows vertically
        within a tolerance.

        Parameters
        ----------
        text : list
            List of PDFMiner text objects.
        row_tol : int, optional (default: 2)

        Returns
        -------
        rows : list
            Two-dimensional list of text objects grouped into rows.

        """
        row_y = None
        rows = []
        temp = []
        non_empty_text = [t for t in text if t.get_text().strip()]
        for t in non_empty_text:
            # is checking for upright necessary?
            # if t.get_text().strip() and all([obj.upright \
            #   for obj in t._objs
            # if type(obj) is LTChar]):
            if row_y is None:
                row_y = t.y0
            elif not np.isclose(row_y, t.y0, atol=row_tol):
                rows.append(sorted(temp, key=lambda t: t.x0))
                temp = []
                # We update the row's bottom as we go, to be forgiving if there
                # is a gradual change across multiple columns.
                row_y = t.y0
            temp.append(t)
        rows.append(sorted(temp, key=lambda t: t.x0))
        return rows

    @staticmethod
    def _merge_columns(l, column_tol=0):
        """Merges column boundaries horizontally if they overlap
        or lie within a tolerance.

        Parameters
        ----------
        l : list
            List of column x-coordinate tuples.
        column_tol : int, optional (default: 0)

        Returns
        -------
        merged : list
            List of merged column x-coordinate tuples.

        """
        merged = []
        for higher in l:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                if column_tol >= 0:
                    if higher[0] <= lower[1] or np.isclose(
                        higher[0], lower[1], atol=column_tol
                    ):
                        upper_bound = max(lower[1], higher[1])
                        lower_bound = min(lower[0], higher[0])
                        merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
                elif column_tol < 0:
                    if higher[0] <= lower[1]:
                        if np.isclose(higher[0], lower[1],
                                      atol=abs(column_tol)):
                            merged.append(higher)
                        else:
                            upper_bound = max(lower[1], higher[1])
                            lower_bound = min(lower[0], higher[0])
                            merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
        return merged

    @staticmethod
    def _join_rows(rows_grouped, text_y_max, text_y_min):
        """Makes row coordinates continuous. For the row to "touch"
        we split the existing gap between them in half.

        Parameters
        ----------
        rows_grouped : list
            Two-dimensional list of text objects grouped into rows.
        text_y_max : int
        text_y_min : int

        Returns
        -------
        rows : list
            List of continuous row y-coordinate tuples.

        """
        row_boundaries = [
            [
                max(t.y1 for t in r),
                min(t.y0 for t in r)
            ]
            for r in rows_grouped
        ]
        for i in range(0, len(row_boundaries)-1):
            top_row = row_boundaries[i]
            bottom_row = row_boundaries[i+1]
            top_row[1] = bottom_row[0] = (top_row[1] + bottom_row[0]) / 2
        row_boundaries[0][0] = text_y_max
        row_boundaries[-1][1] = text_y_min
        return row_boundaries

    @staticmethod
    def _add_columns(cols, text, row_tol):
        """Adds columns to existing list by taking into account
        the text that lies outside the current column x-coordinates.

        Parameters
        ----------
        cols : list
            List of column x-coordinate tuples.
        text : list
            List of PDFMiner text objects.
        ytol : int

        Returns
        -------
        cols : list
            Updated list of column x-coordinate tuples.

        """
        if text:
            text = Hybrid._group_rows(text, row_tol=row_tol)
            elements = [len(r) for r in text]
            new_cols = [
                (t.x0, t.x1)
                for r in text if len(r) == max(elements)
                for t in r
            ]
            cols.extend(Hybrid._merge_columns(sorted(new_cols)))
        return cols

    @staticmethod
    def _join_columns(cols, text_x_min, text_x_max):
        """Makes column coordinates continuous.

        Parameters
        ----------
        cols : list
            List of column x-coordinate tuples.
        text_x_min : int
        text_y_max : int

        Returns
        -------
        cols : list
            Updated list of column x-coordinate tuples.

        """
        cols = sorted(cols)
        cols = [(cols[i][0] + cols[i - 1][1]) / 2 for i in range(1, len(cols))]
        cols.insert(0, text_x_min)
        cols.append(text_x_max)
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        return cols

    def _validate_columns(self):
        if self.table_areas is not None and self.columns is not None:
            if len(self.table_areas) != len(self.columns):
                raise ValueError("Length of table_areas and columns"
                                 " should be equal")

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

    def _generate_table_bbox(self):
        self.textedges = []
        if self.table_areas is None:
            hor_text = self.horizontal_text
            if self.table_regions is not None:
                # filter horizontal text
                hor_text = []
                for region in self.table_regions:
                    x1, y1, x2, y2 = region.split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    region_text = text_in_bbox(
                        (x1, y2, x2, y1), self.horizontal_text)
                    hor_text.extend(region_text)
            # find tables based on nurminen's detection algorithm
            table_bbox = self._nurminen_table_detection(hor_text)
        else:
            table_bbox = {}
            for area in self.table_areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                table_bbox[(x1, y2, x2, y1)] = None
        self.table_bbox = table_bbox

    def _generate_columns_and_rows(self, table_idx, tk):
        # select elements which lie within table_bbox
        self.t_bbox = text_in_bbox_per_axis(
            tk,
            self.horizontal_text,
            self.vertical_text
        )

        text_x_min, text_y_min, text_x_max, text_y_max = \
            self._text_bbox(self.t_bbox)
        rows_grouped = self._group_rows(
            self.t_bbox["horizontal"], row_tol=self.row_tol)
        rows = self._join_rows(rows_grouped, text_y_max, text_y_min)
        elements = [len(r) for r in rows_grouped]

        if self.columns is not None and self.columns[table_idx] != "":
            # user has to input boundary columns too
            # take (0, pdf_width) by default
            # similar to else condition
            # len can't be 1
            cols = self.columns[table_idx].split(",")
            cols = [float(c) for c in cols]
            cols.insert(0, text_x_min)
            cols.append(text_x_max)
            cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        else:
            # calculate mode of the list of number of elements in
            # each row to guess the number of columns
            ncols = max(set(elements), key=elements.count)
            if ncols == 1:
                # if mode is 1, the page usually contains not tables
                # but there can be cases where the list can be skewed,
                # try to remove all 1s from list in this case and
                # see if the list contains elements, if yes, then use
                # the mode after removing 1s
                elements = list(filter(lambda x: x != 1, elements))
                if elements:
                    ncols = max(set(elements), key=elements.count)
                else:
                    warnings.warn(
                        "No tables found in table area {}"
                        .format(table_idx + 1)
                    )
            cols = [
                (t.x0, t.x1)
                for r in rows_grouped
                if len(r) == ncols
                for t in r
            ]
            cols = self._merge_columns(
                sorted(cols),
                column_tol=self.column_tol
            )
            inner_text = []
            for i in range(1, len(cols)):
                left = cols[i - 1][1]
                right = cols[i][0]
                inner_text.extend(
                    [
                        t
                        for direction in self.t_bbox
                        for t in self.t_bbox[direction]
                        if t.x0 > left and t.x1 < right
                    ]
                )
            outer_text = [
                t
                for direction in self.t_bbox
                for t in self.t_bbox[direction]
                if t.x0 > cols[-1][1] or t.x1 < cols[0][0]
            ]
            inner_text.extend(outer_text)
            cols = self._add_columns(cols, inner_text, self.row_tol)
            cols = self._join_columns(cols, text_x_min, text_x_max)

        return cols, rows

    def _generate_table(self, table_idx, cols, rows, **kwargs):
        table = self._initialize_new_table(table_idx, cols, rows)
        table = table.set_all_edges()
        table.record_parse_metadata(self)

        # for plotting
        table._bbox = self.table_bbox
        table._segments = None
        table._textedges = self.textedges

        return table

    def extract_tables(self, filename):
        if self._document_has_no_text():
            return []

        # Identify plausible areas within the doc where tables lie,
        # populate table_bbox keys with these areas.
        self._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, bbox in enumerate(
            sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            cols, rows = self._generate_columns_and_rows(table_idx, bbox)
            table = self._generate_table(table_idx, cols, rows)
            table._bbox = bbox
            _tables.append(table)

        return _tables
