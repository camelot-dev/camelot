"""Contains the core functions to parse tables from PDFs."""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile
import zipfile
from operator import itemgetter
from typing import Any
from typing import Iterable
from typing import Iterator

import cv2
import pandas as pd


if sys.version_info >= (3, 11):
    from typing import TypedDict  # pylint: disable=no-name-in-module
    from typing import Unpack
else:
    from typing_extensions import TypedDict, Unpack

from .backends import ImageConversionBackend
from .utils import build_file_path_in_temp_dir
from .utils import get_index_closest_point
from .utils import get_textline_coords


# minimum number of vertical textline intersections for a textedge
# to be considered valid
TEXTEDGE_REQUIRED_ELEMENTS = 4
# padding added to table area on the left, right and bottom
TABLE_AREA_PADDING = 10


HORIZONTAL_ALIGNMENTS = ["left", "right", "middle"]
VERTICAL_ALIGNMENTS = ["top", "bottom", "center"]
ALL_ALIGNMENTS = HORIZONTAL_ALIGNMENTS + VERTICAL_ALIGNMENTS


class TextAlignment:
    """Represents a list of textlines sharing an alignment on a coordinate.

    The alignment can be left/right/middle or top/bottom/center.
    (PDF coordinate space)

    Parameters
    ----------
    coord : float
        coordinate of the initial text edge. Depending on the alignment
        it could be a vertical or horizontal coordinate.
    textline : obj
        the original textline to start the alignment
    align : str
        Name of the alignment (e.g. "left", "top", etc)

    Attributes
    ----------
    coord : float
        The coordinate aligned averaged out across textlines.  It can be along
        the x or y axis.
    textlines : array
        Array of textlines that demonstrate this alignment.
    align : str
        Name of the alignment (e.g. "left", "top", etc)
    """

    def __init__(self, coord, textline, align):
        self.coord = coord
        self.textlines = [textline]
        self.align = align

    def __repr__(self):  # noqa D105
        text_inside = " | ".join(
            map(lambda x: x.get_text(), self.textlines[:2])
        ).replace("\n", "")
        return (
            f"<TextEdge coord={self.coord} tl={len(self.textlines)} "
            f"textlines text='{text_inside}...'>"
        )

    def register_aligned_textline(self, textline, coord):
        """Update new textline to this alignment, adapting its average."""
        # Increase the intersections for this segment, expand it up,
        # and adjust the x based on the new value
        self.coord = (self.coord * len(self.textlines) + coord) / float(
            len(self.textlines) + 1
        )
        self.textlines.append(textline)


class TextEdge(TextAlignment):
    """Defines a text edge coordinates relative to a left-bottom origin.

    (PDF coordinate space)
    An edge is an alignment bounded over a segment.

    Parameters
    ----------
    coord : float
        coordinate of the text edge.  Can be x or y.
    y0 : float
        y-coordinate of bottommost point.
    y1 : float
        y-coordinate of topmost point.
    align : string, optional (default: 'left')
        {'left', 'right', 'middle'}

    Attributes
    ----------
    is_valid: bool
        A text edge is valid if it intersects with at least
        TEXTEDGE_REQUIRED_ELEMENTS horizontal text rows.

    """

    def __init__(self, coord, textline, align):
        super().__init__(coord, textline, align)
        self.y0 = textline.y0
        self.y1 = textline.y1
        self.is_valid = False

    def __repr__(self):  # noqa D105
        x = round(self.coord, 2)
        y0 = round(self.y0, 2)
        y1 = round(self.y1, 2)
        return (
            f"<TextEdge x={x} y0={y0} y1={y1} align={self.align} valid={self.is_valid}>"
        )

    def update_coords(self, x, textline, edge_tol=50):
        """Update text edge coordinates.

        Update the text edge's x and bottom y coordinates and sets
        the is_valid attribute.
        """
        if math.isclose(self.y0, textline.y0, abs_tol=edge_tol):
            self.register_aligned_textline(textline, x)
            self.y0 = textline.y0
            # a textedge is valid only if it extends uninterrupted
            # over a required number of textlines
            if len(self.textlines) > TEXTEDGE_REQUIRED_ELEMENTS:
                self.is_valid = True


class TextAlignments:
    """Defines a dict of text edges across reference alignments."""

    def __init__(self, alignment_names):
        # For each possible alignment, list of tuples coordinate/textlines
        self._text_alignments = {}
        for alignment_name in alignment_names:
            self._text_alignments[alignment_name] = []

    @staticmethod
    def _create_new_text_alignment(coord, textline, align):
        return TextAlignment(coord, textline, align)

    def _update_alignment(self, alignment, coord, textline):
        return NotImplemented

    def _register_textline(self, textline):
        """Update an existing text edge in the current dict."""
        coords = get_textline_coords(textline)
        for alignment_id, alignment_array in self._text_alignments.items():
            coord = coords[alignment_id]

            # Find the index of the closest existing element (or 0 if none)
            idx_closest = get_index_closest_point(
                coord, alignment_array, fn=lambda x: x.coord
            )

            # Check if the edges before/after are close enough
            # that it can be considered aligned
            idx_insert = None
            if idx_closest is None:
                idx_insert = 0
            else:
                coord_closest = alignment_array[idx_closest].coord
                # Note: np.isclose is slow!
                if coord - 0.5 < coord_closest < coord + 0.5:
                    self._update_alignment(
                        alignment_array[idx_closest], coord, textline
                    )
                elif coord_closest < coord:
                    idx_insert = idx_closest + 1
                else:
                    idx_insert = idx_closest
            if idx_insert is not None:
                new_alignment = self._create_new_text_alignment(
                    coord, textline, alignment_id
                )
                alignment_array.insert(idx_insert, new_alignment)


class TextEdges(TextAlignments):
    """Defines a dict text edges on the PDF page.

    The dict contains the left, right and middle text edges found on
    the PDF page. The dict has three keys based on the alignments,
    and each key's value is a list of camelot.core.TextEdge objects.
    """

    def __init__(self, edge_tol=50):
        super().__init__(HORIZONTAL_ALIGNMENTS)
        self.edge_tol = edge_tol

    def _create_new_text_alignment(self, coord, textline, align):
        # In TextEdges, each alignment is a TextEdge
        return TextEdge(coord, textline, align)

    def add(self, coord, textline, align):
        """Add a new text edge to the current dict."""
        te = self._create_new_text_alignment(coord, textline, align)
        self._text_alignments[align].append(te)

    def _update_alignment(self, alignment, coord, textline):
        alignment.update_coords(coord, textline, self.edge_tol)

    def generate(self, textlines):
        """Generates the text edges dict based on horizontal text rows."""
        for tl in textlines:
            if len(tl.get_text().strip()) > 1:  # TODO: hacky
                self._register_textline(tl)

    def get_relevant(self):
        """Return the list of relevant text edges.

        (all share the same alignment)
        based on which list intersects horizontal text rows the most.
        """
        intersections_sum = {
            "left": sum(
                len(te.textlines) for te in self._text_alignments["left"] if te.is_valid
            ),
            "right": sum(
                len(te.textlines)
                for te in self._text_alignments["right"]
                if te.is_valid
            ),
            "middle": sum(
                len(te.textlines)
                for te in self._text_alignments["middle"]
                if te.is_valid
            ),
        }

        # TODO: naive
        # get vertical textedges that intersect maximum number of
        # times with horizontal textlines
        relevant_align = max(intersections_sum.items(), key=itemgetter(1))[0]
        return list(
            filter(lambda te: te.is_valid, self._text_alignments[relevant_align])
        )

    def get_table_areas(self, textlines, relevant_textedges):
        """
        Return a dict of interesting table areas on the PDF page.

        The table areas are calculated using relevant text edges.

        Parameters
        ----------
        textlines : list
            List of text line objects that are relevant for determining table areas.
        relevant_textedges : list
            List of relevant text edge objects used to identify table areas.

        Returns
        -------
        dict
            A dictionary with padded table areas as keys and None as values.
        """
        # Sort relevant text edges in reading order
        relevant_textedges.sort(key=lambda te: (-te.y0, te.coord))

        table_areas = self._initialize_table_areas(relevant_textedges)
        self._extend_table_areas_with_textlines(table_areas, textlines)

        # Add padding to table areas
        average_textline_height = self._calculate_average_textline_height(textlines)
        padded_table_areas = {
            self._pad(area, average_textline_height): None for area in table_areas
        }

        return padded_table_areas

    def _initialize_table_areas(self, relevant_textedges):
        """
        Initialize table areas based on relevant text edges.

        Parameters
        ----------
        relevant_textedges : list
            List of relevant text edge objects used to initialize table areas.

        Returns
        -------
        dict
            A dictionary of table areas initialized from relevant text edges.
        """
        table_areas = {}
        for te in relevant_textedges:
            if not table_areas:
                table_areas[(te.coord, te.y0, te.coord, te.y1)] = None
            else:
                self._update_table_areas(table_areas, te)

        return table_areas

    def _update_table_areas(self, table_areas, te):
        """
        Update table areas by checking for overlaps with new text edges.

        Parameters
        ----------
        table_areas : dict
            Current table areas to be updated.
        te : object
            The new text edge object to check for overlaps.

        Returns
        -------
        None
        """
        found = None
        for area in table_areas:
            # Check for overlap
            if te.y1 >= area[1] and te.y0 <= area[3]:
                found = area
                break

        if found is None:
            table_areas[(te.coord, te.y0, te.coord, te.y1)] = None
        else:
            table_areas.pop(found)
            updated_area = (
                found[0],
                min(te.y0, found[1]),
                max(found[2], te.coord),
                max(found[3], te.y1),
            )
            table_areas[updated_area] = None

    def _extend_table_areas_with_textlines(self, table_areas, textlines):
        """
        Extend table areas based on text lines that overlap vertically.

        Parameters
        ----------
        table_areas : dict
            Current table areas to be extended.
        textlines : list
            List of text line objects relevant for extending table areas.

        Returns
        -------
        None
        """
        for tl in textlines:
            found = None
            for area in table_areas:
                # Check for overlap
                if tl.y0 >= area[1] and tl.y1 <= area[3]:
                    found = area
                    break

            if found is not None:
                table_areas.pop(found)
                updated_area = (
                    min(tl.x0, found[0]),
                    min(tl.y0, found[1]),
                    max(found[2], tl.x1),
                    max(found[3], tl.y1),
                )
                table_areas[updated_area] = None

    def _calculate_average_textline_height(self, textlines):
        """
        Calculate the average height of text lines.

        Parameters
        ----------
        textlines : list
            List of text line objects.

        Returns
        -------
        float
            The average height of the text lines.
        """
        sum_textline_height = sum(tl.y1 - tl.y0 for tl in textlines)
        return sum_textline_height / float(len(textlines)) if textlines else 0

    def _pad(self, area, average_row_height):
        """
        Pad a given area by a constant value.

        Parameters
        ----------
        area : tuple
            The area to be padded defined as (x0, y0, x1, y1).
        average_row_height : float
            The average height of rows to use for padding.

        Returns
        -------
        tuple
            The padded area.
        """
        x0 = area[0] - TABLE_AREA_PADDING
        y0 = area[1] - TABLE_AREA_PADDING
        x1 = area[2] + TABLE_AREA_PADDING
        # Add a constant since table headers can be relatively up
        y1 = area[3] + average_row_height * 5
        return (x0, y0, x1, y1)


class Cell:
    """Defines a cell in a table.

    With coordinates relative to a
    left-bottom origin. (PDF coordinate space)

    Parameters
    ----------
    x1 : float
        x-coordinate of left-bottom point.
    y1 : float
        y-coordinate of left-bottom point.
    x2 : float
        x-coordinate of right-top point.
    y2 : float
        y-coordinate of right-top point.

    Attributes
    ----------
    lb : tuple
        Tuple representing left-bottom coordinates.
    lt : tuple
        Tuple representing left-top coordinates.
    rb : tuple
        Tuple representing right-bottom coordinates.
    rt : tuple
        Tuple representing right-top coordinates.
    left : bool
        Whether or not cell is bounded on the left.
    right : bool
        Whether or not cell is bounded on the right.
    top : bool
        Whether or not cell is bounded on the top.
    bottom : bool
        Whether or not cell is bounded on the bottom.
    text : string
        Text assigned to cell.

    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.lb = (x1, y1)
        self.lt = (x1, y2)
        self.rb = (x2, y1)
        self.rt = (x2, y2)
        self.left = False
        self.right = False
        self.top = False
        self.bottom = False
        self._text = ""

    def __repr__(self):  # noqa D105
        x1 = round(self.x1)
        y1 = round(self.y1)
        x2 = round(self.x2)
        y2 = round(self.y2)
        return f"<Cell x1={x1} y1={y1} x2={x2} y2={y2}>"

    @property
    def text(self):  # noqa D102
        return self._text

    @text.setter
    def text(self, t):  # noqa D105
        self._text = "".join([self._text, t])

    @property
    def hspan(self) -> bool:
        """Whether or not cell spans horizontally."""
        return not self.left or not self.right

    @property
    def vspan(self) -> bool:
        """Whether or not cell spans vertically."""
        return not self.top or not self.bottom

    @property
    def bound(self):
        """The number of sides on which the cell is bounded."""
        return self.top + self.bottom + self.left + self.right


class Table:
    """Defines a table with coordinates relative to a left-bottom origin.

    (PDF coordinate space)

    Parameters
    ----------
    cols : list
        List of tuples representing column x-coordinates in increasing
        order.
    rows : list
        List of tuples representing row y-coordinates in decreasing
        order.

    Attributes
    ----------
    df : :class:`pandas.DataFrame`
    shape : tuple
        Shape of the table.
    accuracy : float
        Accuracy with which text was assigned to the cell.
    whitespace : float
        Percentage of whitespace in the table.
    filename : str
        Path of the original PDF
    order : int
        Table number on PDF page.
    page : int
        PDF page number.

    """

    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.cells = [[Cell(c[0], r[1], c[1], r[0]) for c in cols] for r in rows]
        self.df = pd.DataFrame()
        self.shape = (0, 0)
        self.accuracy = 0
        self.whitespace = 0
        self.filename = None
        self.order = None
        self.page = None
        self.flavor = None  # Flavor of the parser that generated the table
        self.pdf_size = None  # Dimensions of the original PDF page
        self._bbox = None  # Bounding box in original document
        self.parse = None  # Parse information
        self.parse_details = None  # Field holding debug data

        self._image = None
        self._image_path = None  # Temporary file to hold an image of the pdf

    def __repr__(self):
        """Return a string representation of the class .

        Returns
        -------
        [type]
            [description]
        """
        return f"<{self.__class__.__name__} shape={self.shape}>"

    def __lt__(self, other):
        """Return True if the two pages are less than the current page .

        Parameters
        ----------
        other : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        if self.page == other.page:
            if self.order < other.order:
                return True
        if self.page < other.page:
            return True

    @property
    def data(self):
        """Returns two-dimensional list of strings in table."""
        d = []
        for row in self.cells:
            d.append([cell.text.strip() for cell in row])
        return d

    @property
    def parsing_report(self):
        """Returns a parsing report.

        with % accuracy, % whitespace,
        table number on page and page number.
        """
        # pretty?
        report = {
            "accuracy": round(self.accuracy, 2),
            "whitespace": round(self.whitespace, 2),
            "order": self.order,
            "page": self.page,
        }
        return report

    def get_pdf_image(self):
        """Compute pdf image and cache it."""
        if self._image is None:
            if self._image_path is None:
                self._image_path = build_file_path_in_temp_dir(
                    os.path.basename(self.filename), ".png"
                )
                backend = ImageConversionBackend(use_fallback=True)
                backend.convert(self.filename, self._image_path)
            self._image = cv2.imread(self._image_path)
        return self._image

    def set_all_edges(self):
        """Set all table edges to True."""
        for row in self.cells:
            for cell in row:
                cell.left = cell.right = cell.top = cell.bottom = True
        return self

    def set_edges(self, vertical, horizontal, joint_tol=2):
        """Set the edges of the joint.

        Set a cell's edges to True depending on whether the cell's
        coordinates overlap with the line's coordinates within a
        tolerance.

        Parameters
        ----------
        vertical : list
            List of detected vertical lines.
        horizontal : list
            List of detected horizontal lines.
        joint_tol : int, optional
            Tolerance for determining proximity, by default 2
        """
        self._set_vertical_edges(vertical, joint_tol)
        self._set_horizontal_edges(horizontal, joint_tol)
        return self

    def _find_close_point(self, coords, coord, joint_tol):
        for i, t in enumerate(coords):
            if math.isclose(coord, t[0], abs_tol=joint_tol):
                return i
        return None

    def _set_vertical_edges(self, vertical, joint_tol):
        for v in vertical:
            # find closest x coord
            # iterate over y coords and find closest start and end points
            start = self._find_close_point(self.rows, v[3], joint_tol)
            if start is None:
                continue
            end = self._find_close_point(self.rows, v[1], joint_tol)
            if end is None:
                end = len(self.rows)
            i = self._find_close_point(self.cols, v[0], joint_tol)
            self._update_vertical_edges(start, end, i)

    def _update_vertical_edges(self, start, end, index):
        if index is None:  # only right edge
            index = len(self.cols) - 1
            if index >= 0:
                for j in range(start, end):
                    self.cells[j][index].right = True
        elif index == 0:  # only left edge
            for j in range(start, end):
                self.cells[j][0].left = True
        else:  # both left and right edges
            for j in range(start, end):
                self.cells[j][index].left = True
                self.cells[j][index - 1].right = True

    def _set_horizontal_edges(self, horizontal, joint_tol):
        for h in horizontal:
            # find closest y coord
            # iterate over x coords and find closest start and end points
            start = self._find_close_point(self.cols, h[0], joint_tol)
            if start is None:
                continue
            end = self._find_close_point(self.cols, h[2], joint_tol)
            if end is None:
                end = len(self.cols)
            i = self._find_close_point(self.rows, h[1], joint_tol)
            self._update_horizontal_edges(start, end, i)

    def _update_horizontal_edges(self, start, end, index):
        if index is None:  # only bottom edge
            index = len(self.rows) - 1
            if index >= 0:
                for j in range(start, end):
                    self.cells[index][j].bottom = True
        elif index == 0:  # only top edge
            for j in range(start, end):
                self.cells[0][j].top = True
        else:  # both top and bottom edges
            for j in range(start, end):
                self.cells[index][j].top = True
                self.cells[index - 1][j].bottom = True

    def set_border(self):
        """Sets table border edges to True."""
        num_rows = len(self.rows)
        num_cols = len(self.cols)

        # Ensure cells structure is valid
        if num_rows == 0 or num_cols == 0:
            return self  # No rows or columns, nothing to do

        # Check if cells have the expected structure
        if len(self.cells) != num_rows or any(
            len(row) != num_cols for row in self.cells
        ):
            raise ValueError(
                "Inconsistent cells structure: cells should match the dimensions of rows and cols."
            )

        # Set left and right borders for each row
        for row_index in range(num_rows):
            self.cells[row_index][0].left = True  # Set the left border
            self.cells[row_index][num_cols - 1].right = True  # Set the right border

        # Set top and bottom borders for each column
        for col_index in range(num_cols):
            self.cells[0][col_index].top = True  # Set the top border
            self.cells[num_rows - 1][col_index].bottom = True  # Set the bottom border

        return self

    def copy_spanning_text(self, copy_text=None):
        """
        Copies over text in empty spanning cells.

        Parameters
        ----------
        copy_text : list of str, optional (default: None)
            Select one or more of the following strings: {'h', 'v'} to specify
            the direction in which text should be copied over when a cell spans
            multiple rows or columns.

        Returns
        -------
        camelot.core.Table
            The updated table with copied text in spanning cells.
        """
        if copy_text is None:
            return self

        for direction in copy_text:
            if direction == "h":
                self._copy_horizontal_text()
            elif direction == "v":
                self._copy_vertical_text()

        return self

    def _copy_horizontal_text(self):
        """
        Copies text horizontally in empty spanning cells.

        This method iterates through the cells and fills empty cells that span
        horizontally with the text from the left adjacent cell.

        Returns
        -------
        None
        """
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if (
                    self.cells[i][j].text.strip() == ""
                    and self.cells[i][j].hspan
                    and not self.cells[i][j].left
                ):
                    self.cells[i][j].text = self.cells[i][j - 1].text

    def _copy_vertical_text(self):
        """
        Copies text vertically in empty spanning cells.

        This method iterates through the cells and fills empty cells that span
        vertically with the text from the top adjacent cell.

        Returns
        -------
        None
        """
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if (
                    self.cells[i][j].text.strip() == ""
                    and self.cells[i][j].vspan
                    and not self.cells[i][j].top
                ):
                    self.cells[i][j].text = self.cells[i - 1][j].text

    def to_csv(self, path, **kwargs):
        """Write Table(s) to a comma-separated values (csv) file.

        For kwargs, check :meth:`pandas.DataFrame.to_csv`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8", "index": False, "header": False, "quoting": 1}
        kw.update(kwargs)
        self.df.to_csv(path, **kw)

    def to_json(self, path, **kwargs):
        """Write Table(s) to a JSON file.

        For kwargs, check :meth:`pandas.DataFrame.to_json`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"orient": "records"}
        kw.update(kwargs)
        json_string = self.df.to_json(**kw)
        with open(path, "w") as f:
            f.write(json_string)

    def to_excel(self, path, **kwargs):
        """Write Table(s) to an Excel file.

        For kwargs, check :meth:`pandas.DataFrame.to_excel`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8"}
        sheet_name = f"page-{self.page}-table-{self.order}"
        kw.update(kwargs)
        writer = pd.ExcelWriter(path)
        self.df.to_excel(writer, sheet_name=sheet_name, **kw)

    def to_html(self, path, **kwargs):
        """Write Table(s) to an HTML file.

        For kwargs, check :meth:`pandas.DataFrame.to_html`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        html_string = self.df.to_html(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_string)

    def to_markdown(self, path, **kwargs):
        """Write Table(s) to a Markdown file.

        For kwargs, check :meth:`pandas.DataFrame.to_markdown`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        md_string = self.df.to_markdown(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_string)

    def to_sqlite(self, path, **kwargs):
        """Write Table(s) to sqlite database.

        For kwargs, check :meth:`pandas.DataFrame.to_sql`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"if_exists": "replace", "index": False}
        kw.update(kwargs)
        conn = sqlite3.connect(path)
        table_name = f"page-{self.page}-table-{self.order}"
        self.df.to_sql(table_name, conn, **kw)
        conn.commit()
        conn.close()


class _Kw(TypedDict):
    """Helper class to define file related arguments."""

    path: os.PathLike[Any] | str
    dirname: str
    root: str
    ext: str


class TableList:
    """Defines a list of camelot.core.Table objects.

    Each table can be accessed using its index.

    Attributes
    ----------
    n : int
        Number of tables in the list.

    """

    def __init__(self, tables: Iterable[Table]) -> None:  # noqa D105
        self._tables: Iterable[Table] = tables

    def __repr__(self):  # noqa D105
        return f"<{self.__class__.__name__} n={self.n}>"

    def __len__(self):  # noqa D105
        return len(self._tables)

    def __getitem__(self, idx):  # noqa D105
        return self._tables[idx]

    def __iter__(self) -> Iterator[Table]:  # noqa D105
        return iter(self._tables)

    def __next__(self) -> Table:  # noqa D105
        return next(self)

    @staticmethod
    def _format_func(table, f):
        return getattr(table, f"to_{f}")

    @property
    def n(self) -> int:
        """The number of tables in the list."""
        return len(self)

    def _write_file(self, f=None, **kwargs: Unpack[_Kw]) -> None:
        dirname = kwargs["dirname"]
        root = kwargs["root"]
        ext = kwargs["ext"]
        for table in self._tables:
            filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
            filepath = os.path.join(dirname, filename)
            to_format = self._format_func(table, f)
            to_format(filepath)

    def _compress_dir(self, **kwargs: Unpack[_Kw]) -> None:
        path = kwargs["path"]
        dirname = kwargs["dirname"]
        root = kwargs["root"]
        ext = kwargs["ext"]
        zipname = os.path.join(os.path.dirname(path), root) + ".zip"
        with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
            for table in self._tables:
                filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
                filepath = os.path.join(dirname, filename)
                z.write(filepath, os.path.basename(filepath))

    def export(self, path: str, f="csv", compress=False):
        """Export the list of tables to specified file format.

        Parameters
        ----------
        path : str
            Output filepath.
        f : str
            File format. Can be csv, excel, html, json, markdown or sqlite.
        compress : bool
            Whether or not to add files to a ZIP archive.

        """
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        root, ext = os.path.splitext(basename)
        if compress:
            dirname = tempfile.mkdtemp()

        kwargs: _Kw = {"path": path, "dirname": dirname, "root": root, "ext": ext}

        if f in ["csv", "html", "json", "markdown"]:
            self._write_file(f=f, **kwargs)
            if compress:
                self._compress_dir(**kwargs)
        elif f == "excel":
            filepath = os.path.join(dirname, basename)
            writer = pd.ExcelWriter(filepath)
            for table in self._tables:
                sheet_name = f"page-{table.page}-table-{table.order}"
                table.df.to_excel(writer, sheet_name=sheet_name)
            writer.close()
            if compress:
                zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                    z.write(filepath, os.path.basename(filepath))
        elif f == "sqlite":
            filepath = os.path.join(dirname, basename)
            for table in self._tables:
                table.to_sqlite(filepath)
            if compress:
                zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                    z.write(filepath, os.path.basename(filepath))
