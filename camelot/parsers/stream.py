"""Implementation of the Stream table parser."""

import warnings

from ..core import TextEdges
from ..utils import bbox_from_str
from ..utils import bbox_from_textlines
from ..utils import text_in_bbox
from ..utils import text_in_bbox_per_axis
from .base import TextBaseParser


class Stream(TextBaseParser):
    """Stream method of parsing looks for spaces between text to parse the table.

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
        super().__init__(
            "stream",
            table_regions=table_regions,
            table_areas=table_areas,
            columns=columns,
            # _validate_columns()
            split_text=split_text,
            flag_size=flag_size,
            strip_text=strip_text,
            edge_tol=edge_tol,
            row_tol=row_tol,
            column_tol=column_tol,
        )
        self.textedges = []

    def _nurminen_table_detection(self, textlines):
        """Anssi Nurminen's Table detection algorithm.

        A general implementation of the table detection algorithm
        described by Anssi Nurminen's master's thesis.
        Link: https://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

        Assumes that tables are situated relatively far apart
        vertically.
        """
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
        """Record data about the origin of the table."""
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
                        bbox_from_str(region_str), self.horizontal_text
                    )
                hor_text.extend(region_text)
            # find tables based on nurminen's detection algorithm
            table_bbox_parses = self._nurminen_table_detection(hor_text)
        else:
            table_bbox_parses = {}
            for area_str in self.table_areas:
                table_bbox_parses[bbox_from_str(area_str)] = None
        self.table_bbox_parses = table_bbox_parses

    def _generate_columns_and_rows(self, bbox, user_cols):
        # select elements which lie within table_bbox
        self.t_bbox = text_in_bbox_per_axis(
            bbox, self.horizontal_text, self.vertical_text
        )

        text_x_min, text_y_min, text_x_max, text_y_max = bbox_from_textlines(
            self.t_bbox["horizontal"] + self.t_bbox["vertical"]
        )

        rows_grouped = self._group_rows(self.t_bbox["horizontal"], row_tol=self.row_tol)
        rows = self._join_rows(rows_grouped, text_y_max, text_y_min)
        elements = [len(r) for r in rows_grouped]

        if user_cols is not None:
            cols = [text_x_min] + user_cols + [text_x_max]
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
                    if elements:
                        ncols = max(set(elements), key=elements.count)
                    else:
                        warnings.warn(
                            f"No tables found in table area {bbox}", stacklevel=2
                        )
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
        return cols, rows, None, None
