# -*- coding: utf-8 -*-

import os
import sqlite3
import zipfile
import tempfile
from operator import itemgetter

import numpy as np
import pandas as pd

from cv2 import cv2

from .utils import (
    get_index_closest_point,
    get_textline_coords,
    build_file_path_in_temp_dir,
    export_pdf_as_png
)


# minimum number of vertical textline intersections for a textedge
# to be considered valid
TEXTEDGE_REQUIRED_ELEMENTS = 4
# padding added to table area on the left, right and bottom
TABLE_AREA_PADDING = 10


HORIZONTAL_ALIGNMENTS = ["left", "right", "middle"]
VERTICAL_ALIGNMENTS = ["top", "bottom", "center"]
ALL_ALIGNMENTS = HORIZONTAL_ALIGNMENTS + VERTICAL_ALIGNMENTS


class TextAlignment():
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

    def __repr__(self):
        text_inside = " | ".join(
            map(lambda x: x.get_text(), self.textlines[:2])).replace("\n", "")
        return f"<TextEdge coord={self.coord} tl={len(self.textlines)} " \
               f"textlines text='{text_inside}...'>"

    def register_aligned_textline(self, textline, coord):
        """Update new textline to this alignment, adapting its average."""
        # Increase the intersections for this segment, expand it up,
        # and adjust the x based on the new value
        self.coord = (self.coord * len(self.textlines) + coord) / \
            float(len(self.textlines) + 1)
        self.textlines.append(textline)


class TextEdge(TextAlignment):
    """Defines a text edge coordinates relative to a left-bottom
    origin. (PDF coordinate space).

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

    def __repr__(self):
        x = round(self.coord, 2)
        y0 = round(self.y0, 2)
        y1 = round(self.y1, 2)
        return f"<TextEdge x={x} y0={y0} y1={y1} align={self.align} " \
            f"valid={self.is_valid}>"

    def update_coords(self, x, textline, edge_tol=50):
        """Updates the text edge's x and bottom y coordinates and sets
        the is_valid attribute.
        """
        if np.isclose(self.y0, textline.y0, atol=edge_tol):
            self.register_aligned_textline(textline, x)
            self.y0 = textline.y0
            # a textedge is valid only if it extends uninterrupted
            # over a required number of textlines
            if len(self.textlines) > TEXTEDGE_REQUIRED_ELEMENTS:
                self.is_valid = True


class TextAlignments():
    """Defines a dict of text edges across reference alignments.
    """

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
        """Updates an existing text edge in the current dict.
        """
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
                        alignment_array[idx_closest],
                        coord,
                        textline
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
    """Defines a dict of left, right and middle text edges found on
    the PDF page. The dict has three keys based on the alignments,
    and each key's value is a list of camelot.core.TextEdge objects.
    """

    def __init__(self, edge_tol=50):
        super().__init__(HORIZONTAL_ALIGNMENTS)
        self.edge_tol = edge_tol

    @staticmethod
    def _create_new_text_alignment(coord, textline, align):
        # In TextEdges, each alignment is a TextEdge
        return TextEdge(coord, textline, align)

    def add(self, coord, textline, align):
        """Adds a new text edge to the current dict."""
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
        """Returns the list of relevant text edges (all share the same
        alignment) based on which list intersects horizontal text rows
        the most.
        """
        intersections_sum = {
            "left": sum(
                len(te.textlines) for te in self._text_alignments["left"]
                if te.is_valid
            ),
            "right": sum(
                len(te.textlines) for te in self._text_alignments["right"]
                if te.is_valid
            ),
            "middle": sum(
                len(te.textlines) for te in self._text_alignments["middle"]
                if te.is_valid
            ),
        }

        # TODO: naive
        # get vertical textedges that intersect maximum number of
        # times with horizontal textlines
        relevant_align = max(intersections_sum.items(), key=itemgetter(1))[0]
        return list(filter(
            lambda te: te.is_valid,
            self._text_alignments[relevant_align])
        )

    def get_table_areas(self, textlines, relevant_textedges):
        """Returns a dict of interesting table areas on the PDF page
        calculated using relevant text edges.
        """

        def pad(area, average_row_height):
            x0 = area[0] - TABLE_AREA_PADDING
            y0 = area[1] - TABLE_AREA_PADDING
            x1 = area[2] + TABLE_AREA_PADDING
            # add a constant since table headers can be relatively up
            y1 = area[3] + average_row_height * 5
            return (x0, y0, x1, y1)

        # sort relevant textedges in reading order
        relevant_textedges.sort(key=lambda te: (-te.y0, te.coord))

        table_areas = {}
        for te in relevant_textedges:
            if not table_areas:
                table_areas[(te.coord, te.y0, te.coord, te.y1)] = None
            else:
                found = None
                for area in table_areas:
                    # check for overlap
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

        # extend table areas based on textlines that overlap
        # vertically. it's possible that these textlines were
        # eliminated during textedges generation since numbers and
        # chars/words/sentences are often aligned differently.
        # drawback: table areas that have paragraphs on their sides
        # will include the paragraphs too.
        sum_textline_height = 0
        for tl in textlines:
            sum_textline_height += tl.y1 - tl.y0
            found = None
            for area in table_areas:
                # check for overlap
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
        average_textline_height = sum_textline_height / \
            float(len(textlines))

        # add some padding to table areas
        table_areas_padded = {}
        for area in table_areas:
            table_areas_padded[pad(area, average_textline_height)] = None

        return table_areas_padded


class Cell():
    """Defines a cell in a table with coordinates relative to a
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
    hspan : bool
        Whether or not cell spans horizontally.
    vspan : bool
        Whether or not cell spans vertically.
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
        self.hspan = False
        self.vspan = False
        self._text = ""

    def __repr__(self):
        x1 = round(self.x1, 2)
        y1 = round(self.y1, 2)
        x2 = round(self.x2, 2)
        y2 = round(self.y2, 2)
        return f"<Cell x1={x1} y1={y1} x2={x2} y2={y2}>"

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, t):
        self._text = "".join([self._text, t])

    @property
    def bound(self):
        """The number of sides on which the cell is bounded."""
        return self.top + self.bottom + self.left + self.right


class Table():
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
        self.cells = [
            [Cell(c[0], r[1], c[1], r[0]) for c in cols] for r in rows
        ]
        self.df = None
        self.shape = (0, 0)
        self.accuracy = 0
        self.whitespace = 0
        self.filename = None
        self.order = None
        self.page = None
        self.flavor = None         # Flavor of the parser used
        self.pdf_size = None       # Dimensions of the original PDF page
        self._bbox = None          # Bounding box in original document
        self.parse = None          # Parse information
        self.parse_details = None  # Field holding extra debug data

        self._image = None
        self._image_path = None  # Temporary file to hold an image of the pdf

        self._text = []      # List of text box coordinates
        self.textlines = []  # List of actual textlines on the page

    def __repr__(self):
        return f"<{self.__class__.__name__} shape={self.shape}>"

    def __lt__(self, other):
        if self.page == other.page:
            if self.order < other.order:
                return True
        return self.page < other.page

    @property
    def data(self):
        """Returns two-dimensional list of strings in table.
        """
        d = []
        for row in self.cells:
            d.append([cell.text.strip() for cell in row])
        return d

    @property
    def parsing_report(self):
        """Returns a parsing report with %accuracy, %whitespace,
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
        """Compute pdf image and cache it
        """
        if self._image is None:
            if self._image_path is None:
                self._image_path = build_file_path_in_temp_dir(
                    os.path.basename(self.filename),
                    ".png"
                )
                export_pdf_as_png(self.filename, self._image_path)
            self._image = cv2.imread(self._image_path)
        return self._image

    def set_all_edges(self):
        """Sets all table edges to True.
        """
        for row in self.cells:
            for cell in row:
                cell.left = cell.right = cell.top = cell.bottom = True
        return self

    def set_edges(self, vertical, horizontal, joint_tol=2):
        """Sets a cell's edges to True depending on whether the cell's
        coordinates overlap with the line's coordinates within a
        tolerance.

        Parameters
        ----------
        vertical : list
            List of detected vertical lines.
        horizontal : list
            List of detected horizontal lines.

        """
        for v in vertical:
            # find closest x coord
            # iterate over y coords and find closest start and end points
            i = [
                i
                for i, t in enumerate(self.cols)
                if np.isclose(v[0], t[0], atol=joint_tol)
            ]
            j = [
                j
                for j, t in enumerate(self.rows)
                if np.isclose(v[3], t[0], atol=joint_tol)
            ]
            k = [
                k
                for k, t in enumerate(self.rows)
                if np.isclose(v[1], t[0], atol=joint_tol)
            ]
            if not j:
                continue
            J = j[0]
            if i == [0]:  # only left edge
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].left = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].left = True
                        J += 1
            elif i == []:  # only right edge
                L = len(self.cols) - 1
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].right = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].right = True
                        J += 1
            else:  # both left and right edges
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].left = True
                        self.cells[J][L - 1].right = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].left = True
                        self.cells[J][L - 1].right = True
                        J += 1

        for h in horizontal:
            # find closest y coord
            # iterate over x coords and find closest start and end points
            i = [
                i
                for i, t in enumerate(self.rows)
                if np.isclose(h[1], t[0], atol=joint_tol)
            ]
            j = [
                j
                for j, t in enumerate(self.cols)
                if np.isclose(h[0], t[0], atol=joint_tol)
            ]
            k = [
                k
                for k, t in enumerate(self.cols)
                if np.isclose(h[2], t[0], atol=joint_tol)
            ]
            if not j:
                continue
            J = j[0]
            if i == [0]:  # only top edge
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].top = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].top = True
                        J += 1
            elif i == []:  # only bottom edge
                L = len(self.rows) - 1
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].bottom = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].bottom = True
                        J += 1
            else:  # both top and bottom edges
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].top = True
                        self.cells[L - 1][J].bottom = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].top = True
                        self.cells[L - 1][J].bottom = True
                        J += 1

        return self

    def set_border(self):
        """Sets table border edges to True.
        """
        for r in range(len(self.rows)):
            self.cells[r][0].left = True
            self.cells[r][len(self.cols) - 1].right = True
        for c in range(len(self.cols)):
            self.cells[0][c].top = True
            self.cells[len(self.rows) - 1][c].bottom = True
        return self

    def set_span(self):
        """Sets a cell's hspan or vspan attribute to True depending
        on whether the cell spans horizontally or vertically.
        """
        for row in self.cells:
            for cell in row:
                left = cell.left
                right = cell.right
                top = cell.top
                bottom = cell.bottom
                if cell.bound == 4:
                    continue
                if cell.bound == 3:
                    if not left and (right and top and bottom):
                        cell.hspan = True
                    elif not right and (left and top and bottom):
                        cell.hspan = True
                    elif not top and (left and right and bottom):
                        cell.vspan = True
                    elif not bottom and (left and right and top):
                        cell.vspan = True
                elif cell.bound == 2:
                    if left and right and (not top and not bottom):
                        cell.vspan = True
                    elif top and bottom and (not left and not right):
                        cell.hspan = True
                elif cell.bound in [0, 1]:
                    cell.vspan = True
                    cell.hspan = True
        return self

    def to_csv(self, path, **kwargs):
        """Writes Table to a comma-separated values (csv) file.

        For kwargs, check :meth:`pandas.DataFrame.to_csv`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8", "index": False, "header": False,
              "quoting": 1}
        kw.update(kwargs)
        self.df.to_csv(path, **kw)

    def to_json(self, path, **kwargs):
        """Writes Table to a JSON file.

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
        """Writes Table to an Excel file.

        For kwargs, check :meth:`pandas.DataFrame.to_excel`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {
            "sheet_name": f"page-{self.page}-table-{self.order}",
            "encoding": "utf-8",
        }
        kw.update(kwargs)
        # pylint: disable=abstract-class-instantiated
        writer = pd.ExcelWriter(path)
        self.df.to_excel(writer, **kw)
        writer.save()

    def to_html(self, path, **kwargs):
        """Writes Table to an HTML file.

        For kwargs, check :meth:`pandas.DataFrame.to_html`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        html_string = self.df.to_html(**kwargs)
        with open(path, "w") as f:
            f.write(html_string)

    def to_sqlite(self, path, **kwargs):
        """Writes Table to sqlite database.

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

    def copy_spanning_text(self, copy_text=None):
        """Copies over text in empty spanning cells.

        Parameters
        ----------
        copy_text : list, optional (default: None)
            {'h', 'v'}
            Select one or more strings from above and pass them as a list
            to specify the direction in which text should be copied over
            when a cell spans multiple rows or columns.

        Returns
        -------
        t : camelot.core.Table

        """
        for f in copy_text:
            if f == "h":
                for i, row in enumerate(self.cells):
                    for j, cell in enumerate(row):
                        if cell.text.strip() == "" and \
                           cell.hspan and \
                           not cell.left:
                            cell.text = self.cells[i][j - 1].text
            elif f == "v":
                for i, row in enumerate(self.cells):
                    for j, cell in enumerate(row):
                        if cell.text.strip() == "" and \
                           cell.vspan and \
                           not cell.top:
                            cell.text = self.cells[i - 1][j].text
        return self


class TableList():
    """Defines a list of camelot.core.Table objects. Each table can
    be accessed using its index.

    Attributes
    ----------
    n : int
        Number of tables in the list.

    """

    def __init__(self, tables):
        self._tables = tables

    def __repr__(self):
        return f"<{self.__class__.__name__} n={self.n}>"

    def __len__(self):
        return len(self._tables)

    def __getitem__(self, idx):
        return self._tables[idx]

    @staticmethod
    def _format_func(table, f):
        return getattr(table, f"to_{f}")

    @property
    def n(self):
        return len(self)

    def _write_file(self, f=None, **kwargs):
        dirname = kwargs.get("dirname")
        root = kwargs.get("root")
        ext = kwargs.get("ext")
        for table in self._tables:
            filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
            filepath = os.path.join(dirname, filename)
            to_format = self._format_func(table, f)
            to_format(filepath)

    def _compress_dir(self, **kwargs):
        path = kwargs.get("path")
        dirname = kwargs.get("dirname")
        root = kwargs.get("root")
        ext = kwargs.get("ext")
        zipname = os.path.join(os.path.dirname(path), root) + ".zip"
        with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
            for table in self._tables:
                filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
                filepath = os.path.join(dirname, filename)
                z.write(filepath, os.path.basename(filepath))

    def export(self, path, f="csv", compress=False):
        """Exports the list of tables to specified file format.

        Parameters
        ----------
        path : str
            Output filepath.
        f : str
            File format. Can be csv, json, excel, html and sqlite.
        compress : bool
            Whether or not to add files to a ZIP archive.

        """
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        root, ext = os.path.splitext(basename)
        if compress:
            dirname = tempfile.mkdtemp()

        kwargs = {"path": path, "dirname": dirname, "root": root, "ext": ext}

        if f in ["csv", "json", "html"]:
            self._write_file(f=f, **kwargs)
            if compress:
                self._compress_dir(**kwargs)
        elif f == "excel":
            filepath = os.path.join(dirname, basename)
            # pylint: disable=abstract-class-instantiated
            writer = pd.ExcelWriter(filepath)
            for table in self._tables:
                sheet_name = f"page-{table.page}-table-{table.order}"
                table.df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    encoding="utf-8"
                )
            writer.save()
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
