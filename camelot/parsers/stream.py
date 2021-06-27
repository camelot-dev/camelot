# -*- coding: utf-8 -*-

import os
import logging
import warnings

import numpy as np
import pandas as pd

from .base import BaseParser
from ..core import TextEdges, Table
from ..utils import text_in_bbox, get_table_index, compute_accuracy, compute_whitespace


logger = logging.getLogger("camelot")


class Stream(BaseParser):
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
        split_text=False,
        flag_size=False,
        strip_text="",
        edge_tol=50,
        row_tol=2,
        column_tol=0,
        **kwargs,
    ):
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.columns = columns
        self._validate_columns()
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
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
        xmin = min([t.x0 for direction in t_bbox for t in t_bbox[direction]])
        ymin = min([t.y0 for direction in t_bbox for t in t_bbox[direction]])
        xmax = max([t.x1 for direction in t_bbox for t in t_bbox[direction]])
        ymax = max([t.y1 for direction in t_bbox for t in t_bbox[direction]])
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
        row_y = 0
        rows = []
        temp = []

        for t in text:
            # is checking for upright necessary?
            # if t.get_text().strip() and all([obj.upright for obj in t._objs if
            # type(obj) is LTChar]):
            if t.get_text().strip():
                if not np.isclose(row_y, t.y0, atol=row_tol):
                    rows.append(sorted(temp, key=lambda t: t.x0))
                    temp = []
                    row_y = t.y0
                temp.append(t)

        rows.append(sorted(temp, key=lambda t: t.x0))
        if len(rows) > 1:
            __ = rows.pop(0)  # TODO: hacky
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
                        if np.isclose(higher[0], lower[1], atol=abs(column_tol)):
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
        """Makes row coordinates continuous.

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
        row_mids = [
            sum([(t.y0 + t.y1) / 2 for t in r]) / len(r) if len(r) > 0 else 0
            for r in rows_grouped
        ]
        rows = [(row_mids[i] + row_mids[i - 1]) / 2 for i in range(1, len(row_mids))]
        rows.insert(0, text_y_max)
        rows.append(text_y_min)
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]
        return rows

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
            text = Stream._group_rows(text, row_tol=row_tol)
            elements = [len(r) for r in text]
            new_cols = [
                (t.x0, t.x1) for r in text if len(r) == max(elements) for t in r
            ]
            cols.extend(Stream._merge_columns(sorted(new_cols)))
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
                raise ValueError("Length of table_areas and columns" " should be equal")

    def _nurminen_table_detection(self, textlines):
        """A general implementation of the table detection algorithm
        described by Anssi Nurminen's master's thesis.
        Link: https://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

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
        if not len(table_bbox):
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
                    region_text = text_in_bbox((x1, y2, x2, y1), self.horizontal_text)
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
        t_bbox = {}
        t_bbox["horizontal"] = text_in_bbox(tk, self.horizontal_text)
        t_bbox["vertical"] = text_in_bbox(tk, self.vertical_text)

        t_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
        t_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))

        self.t_bbox = t_bbox

        text_x_min, text_y_min, text_x_max, text_y_max = self._text_bbox(self.t_bbox)
        rows_grouped = self._group_rows(self.t_bbox["horizontal"], row_tol=self.row_tol)
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
            if not len(elements):
                cols = [(text_x_min, text_x_max)]
            else:
                ncols = max(set(elements), key=elements.count)
                if ncols == 1:
                    # if mode is 1, the page usually contains not tables
                    # but there can be cases where the list can be skewed,
                    # try to remove all 1s from list in this case and
                    # see if the list contains elements, if yes, then use
                    # the mode after removing 1s
                    elements = list(filter(lambda x: x != 1, elements))
                    if len(elements):
                        ncols = max(set(elements), key=elements.count)
                    else:
                        warnings.warn(f"No tables found in table area {table_idx + 1}")
                cols = [
                    (t.x0, t.x1) for r in rows_grouped if len(r) == ncols for t in r
                ]
                cols = self._merge_columns(sorted(cols), column_tol=self.column_tol)
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
        table = Table(cols, rows)
        table = table.set_all_edges()

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
                    for r_idx, c_idx, text in indices:
                        table.cells[r_idx][c_idx].text = text
        accuracy = compute_accuracy([[100, pos_errors]])

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        whitespace = compute_whitespace(data)
        table.flavor = "stream"
        table.accuracy = accuracy
        table.whitespace = whitespace
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table._image = None
        table._segments = None
        table._textedges = self.textedges

        return table

    def extract_tables(self, filename, suppress_stdout=False, layout_kwargs={}):
        self._generate_layout(filename, layout_kwargs)
        base_filename = os.path.basename(self.rootname)

        if not suppress_stdout:
            logger.info(f"Processing {base_filename}")

        if not self.horizontal_text:
            if self.images:
                warnings.warn(
                    f"{base_filename} is image-based, camelot only works on"
                    " text-based pages."
                )
            else:
                warnings.warn(f"No tables found on {base_filename}")
            return []

        self._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, tk in enumerate(
            sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            cols, rows = self._generate_columns_and_rows(table_idx, tk)
            table = self._generate_table(table_idx, cols, rows)
            table._bbox = tk
            _tables.append(table)

        return _tables
