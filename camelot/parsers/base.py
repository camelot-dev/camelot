"""Defines a base parser. As well as generic methods for other parsers."""

import math
import os
import warnings

import pandas as pd

from ..core import Table
from ..utils import bbox_from_str
from ..utils import compute_accuracy
from ..utils import compute_whitespace
from ..utils import get_table_index
from ..utils import text_in_bbox


class BaseParser:
    """Defines a base parser."""

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
        debug=False,
    ):
        self.id = parser_id
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.table_bbox_parses = {}

        self.columns = None
        self.copy_text = copy_text
        self.split_text = split_text
        self.strip_text = strip_text
        self.shift_text = shift_text

        self.flag_size = flag_size

        self.rootname = None
        self.t_bbox = None

        # For plotting details of parsing algorithms
        self.resolution = 300  # default plotting resolution of the PDF.
        self.parse_details = {}
        if not debug:
            self.parse_details = None

    def table_bboxes(self):
        """Return a list of table bounding boxes sorted by position .

        Returns
        -------
        [type]
            [description]
        """
        return sorted(self.table_bbox_parses.keys(), key=lambda x: x[1], reverse=True)

    def prepare_page_parse(
        self,
        filename,
        layout,
        dimensions,
        page_idx,
        images,
        horizontal_text,
        vertical_text,
        layout_kwargs,
    ):
        """Prepare the page for parsing."""
        self.filename = filename
        self.layout_kwargs = layout_kwargs
        self.layout = layout
        self.dimensions = dimensions
        self.page = page_idx
        self.images = images
        self.horizontal_text = horizontal_text
        self.vertical_text = vertical_text
        self.pdf_width, self.pdf_height = self.dimensions
        self.rootname, __ = os.path.splitext(self.filename)

        if self.parse_details is not None:
            self.parse_details["table_regions"] = self.table_regions
            self.parse_details["table_areas"] = self.table_areas

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
                region_text = text_in_bbox(bbox_from_str(region_str), textlines)
                filtered_textlines.extend(region_text)
        return filtered_textlines

    def _document_has_no_text(self):
        """Detect image only documents and warns.

        Returns
        -------
        has_no_text : bool
            Whether the document doesn't have any text at all.
        """
        if not self.horizontal_text:
            rootname = os.path.basename(self.rootname)
            if self.images:
                warnings.warn(
                    f"{rootname} is image-based, "
                    "camelot only works on text-based pages.",
                    stacklevel=1,
                )
            else:
                warnings.warn(f"No tables found on {rootname}", stacklevel=2)
            return True
        return False

    def _initialize_new_table(self, table_idx, bbox, cols, rows):
        """Initialize new table object, ready to be populated.

        Parameters
        ----------
        table_idx : int
            Index of this table within the pdf page analyzed
        bbox : set
            bounding box of this table within the pdf page analyzed
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
        table._bbox = bbox
        return table

    @staticmethod
    def _reduce_index(t, idx, shift_text):
        """
        Reduces index of a text object if it lies within a spanning cell.

        Only useful for some parsers (e.g. Lattice), base method is a
        noop.
        """
        return idx

    def compute_parse_errors(self, table):
        """Compute parse errors for the table .

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        Tuple
            Parse errors
        """
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
                if len(indices) > 0:
                    if indices[0][:2] != (-1, -1):
                        pos_errors.append(error)
                        indices = type(self)._reduce_index(
                            table, indices, shift_text=self.shift_text
                        )
                        for r_idx, c_idx, text in indices:
                            table.cells[r_idx][c_idx].text = text
        return pos_errors

    def _generate_columns_and_rows(self, bbox, user_cols):
        # Pure virtual, must be defined by the derived parser
        raise NotImplementedError()

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        # Pure virtual, must be defined by the derived parser
        raise NotImplementedError()

    def _generate_table_bbox(self):
        # Pure virtual, must be defined by the derived parser
        raise NotImplementedError()

    def extract_tables(self):
        """Extract tables from the document."""
        if self._document_has_no_text():
            return []

        # Identify plausible areas within the doc where tables lie,
        # populate table_bbox keys with these areas.
        self._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, bbox in enumerate(self.table_bboxes()):
            if self.columns is not None and self.columns[table_idx] != "":
                # user has to input boundary columns too
                # take (0, pdf_width) by default
                # similar to else condition
                # len can't be 1
                user_cols = self.columns[table_idx].split(",")
                user_cols = [float(c) for c in user_cols]
            else:
                user_cols = None

            cols, rows, v_s, h_s = self._generate_columns_and_rows(bbox, user_cols)
            table = self._generate_table(table_idx, bbox, cols, rows, v_s=v_s, h_s=h_s)
            _tables.append(table)

        return _tables

    def record_parse_metadata(self, table):
        """Record data about the origin of the table."""
        table.flavor = self.id
        table.filename = self.filename
        if table._bbox in self.table_bbox_parses:
            table.parse = self.table_bbox_parses[table._bbox]
        else:
            # Handle the KeyError gracefully by returning empty lists
            # or by performing alternative logic, such as using a default
            # bounding box or skipping the table.
            print(
                f"Warning: Bounding box {table._bbox} not found in table_bbox_parses."
            )
            return [], [], [], []  # Return empty lists for cols, rows, v_s, h_s
        table.parse_details = self.parse_details
        pos_errors = self.compute_parse_errors(table)
        table.accuracy = compute_accuracy([[100, pos_errors]])

        if self.copy_text is not None:
            table.copy_spanning_text(self.copy_text)

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        table.whitespace = compute_whitespace(data)
        table.pdf_size = (self.pdf_width, self.pdf_height)

        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table.textlines = self.horizontal_text + self.vertical_text


class TextBaseParser(BaseParser):
    """Base class for all text parsers."""

    def __init__(
        self,
        parser_id,
        table_regions=None,
        table_areas=None,
        columns=None,
        flag_size=False,
        split_text=False,
        strip_text="",
        edge_tol=50,
        row_tol=2,
        column_tol=0,
        debug=False,
        **kwargs,
    ):
        """Initialize the text base parser class with default values."""
        super().__init__(
            parser_id,
            table_regions=table_regions,
            table_areas=table_areas,
            split_text=split_text,
            strip_text=strip_text,
            flag_size=flag_size,
            debug=debug,
        )
        self.columns = columns
        self._validate_columns()
        self.edge_tol = edge_tol
        self.row_tol = row_tol
        self.column_tol = column_tol

    @staticmethod
    def _group_rows(text, row_tol=2):
        """
        Group PDFMiner text objects into rows vertically within a tolerance.

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
        text.sort(key=lambda x: (-x.y0, x.x0))
        non_empty_text = [t for t in text if t.get_text().strip()]
        for t in non_empty_text:
            # is checking for upright necessary?
            # if t.get_text().strip() and all([obj.upright \
            #   for obj in t._objs
            # if type(obj) is LTChar]):
            if row_y is None:
                row_y = t.y0
            elif not math.isclose(row_y, t.y0, abs_tol=row_tol):
                rows.append(sorted(temp, key=lambda t: t.x0))
                temp = []
                # We update the row's bottom as we go, to be forgiving if there
                # is a gradual change across multiple columns.
                row_y = t.y0
            temp.append(t)
        rows.append(sorted(temp, key=lambda t: t.x0))
        return rows

    @staticmethod
    def _merge_columns(cl, column_tol=0):
        """Merge column boundaries if they overlap or lie within a tolerance.

        Parameters
        ----------
        cl : list
            List of column x-coordinate tuples.
        column_tol : int, optional (default: 0)

        Returns
        -------
        merged : list
            List of merged column x-coordinate tuples.

        """
        merged = []
        for higher in cl:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                if column_tol >= 0:
                    if higher[0] <= lower[1] or math.isclose(
                        higher[0], lower[1], abs_tol=column_tol
                    ):
                        upper_bound = max(lower[1], higher[1])
                        lower_bound = min(lower[0], higher[0])
                        merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
                elif column_tol < 0:
                    if higher[0] <= lower[1]:
                        if math.isclose(higher[0], lower[1], abs_tol=abs(column_tol)):
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
        """
        Make row coordinates continuous.

        For the row to "touch"
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
            [max(t.y1 for t in r), min(t.y0 for t in r)] for r in rows_grouped
        ]
        for i in range(0, len(row_boundaries) - 1):
            top_row = row_boundaries[i]
            bottom_row = row_boundaries[i + 1]
            top_row[1] = bottom_row[0] = (top_row[1] + bottom_row[0]) / 2
        row_boundaries[0][0] = text_y_max
        row_boundaries[-1][1] = text_y_min
        return row_boundaries

    @staticmethod
    def _add_columns(cols, text, row_tol):
        """Adds columns to existing list.

        By taking into account
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
            text = TextBaseParser._group_rows(text, row_tol=row_tol)
            elements = [len(r) for r in text]
            new_cols = [
                (t.x0, t.x1) for r in text if len(r) == max(elements) for t in r
            ]
            cols.extend(TextBaseParser._merge_columns(sorted(new_cols)))
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

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        table = self._initialize_new_table(table_idx, bbox, cols, rows)
        table = table.set_all_edges()
        self.record_parse_metadata(table)

        return table

    def record_parse_metadata(self, table):
        """Record data about the origin of the table."""
        super().record_parse_metadata(table)
        # for plotting
        table._segments = None
