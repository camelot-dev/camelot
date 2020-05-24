# -*- coding: utf-8 -*-

import os
import sys
import copy
import locale
import logging
import warnings
import subprocess

import numpy as np
import pandas as pd

from .base import BaseParser
from ..core import Table
from ..utils import (
    scale_image,
    scale_pdf,
    segments_in_bbox,
    text_in_bbox,
    merge_close_lines,
    get_table_index,
    compute_accuracy,
    compute_whitespace,
)
from ..image_processing import (
    adaptive_threshold,
    find_lines,
    find_contours,
    find_joints,
)


logger = logging.getLogger("camelot")


class Lattice(BaseParser):
    """Lattice method of parsing looks for lines between text
    to parse the table.

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
    process_background : bool, optional (default: False)
        Process background lines.
    line_scale : int, optional (default: 15)
        Line size scaling factor. The larger the value the smaller
        the detected lines. Making it very large will lead to text
        being detected as lines.
    copy_text : list, optional (default: None)
        {'h', 'v'}
        Direction in which text in a spanning cell will be copied
        over.
    shift_text : list, optional (default: ['l', 't'])
        {'l', 'r', 't', 'b'}
        Direction in which text in a spanning cell will flow.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    line_tol : int, optional (default: 2)
        Tolerance parameter used to merge close vertical and horizontal
        lines.
    joint_tol : int, optional (default: 2)
        Tolerance parameter used to decide whether the detected lines
        and points lie close to each other.
    threshold_blocksize : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    threshold_constant : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    iterations : int, optional (default: 0)
        Number of times for erosion/dilation is applied.

        For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.
    resolution : int, optional (default: 300)
        Resolution used for PDF to PNG conversion.

    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        process_background=False,
        line_scale=15,
        copy_text=None,
        shift_text=["l", "t"],
        split_text=False,
        flag_size=False,
        strip_text="",
        line_tol=2,
        joint_tol=2,
        threshold_blocksize=15,
        threshold_constant=-2,
        iterations=0,
        resolution=300,
        **kwargs
    ):
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.process_background = process_background
        self.line_scale = line_scale
        self.copy_text = copy_text
        self.shift_text = shift_text
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.threshold_blocksize = threshold_blocksize
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution

    @staticmethod
    def _reduce_index(t, idx, shift_text):
        """Reduces index of a text object if it lies within a spanning
        cell.

        Parameters
        ----------
        table : camelot.core.Table
        idx : list
            List of tuples of the form (r_idx, c_idx, text).
        shift_text : list
            {'l', 'r', 't', 'b'}
            Select one or more strings from above and pass them as a
            list to specify where the text in a spanning cell should
            flow.

        Returns
        -------
        indices : list
            List of tuples of the form (r_idx, c_idx, text) where
            r_idx and c_idx are new row and column indices for text.

        """
        indices = []
        for r_idx, c_idx, text in idx:
            for d in shift_text:
                if d == "l":
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].left:
                            c_idx -= 1
                if d == "r":
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].right:
                            c_idx += 1
                if d == "t":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].top:
                            r_idx -= 1
                if d == "b":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].bottom:
                            r_idx += 1
            indices.append((r_idx, c_idx, text))
        return indices

    @staticmethod
    def _copy_spanning_text(t, copy_text=None):
        """Copies over text in empty spanning cells.

        Parameters
        ----------
        t : camelot.core.Table
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
                for i in range(len(t.cells)):
                    for j in range(len(t.cells[i])):
                        if t.cells[i][j].text.strip() == "":
                            if t.cells[i][j].hspan and not t.cells[i][j].left:
                                t.cells[i][j].text = t.cells[i][j - 1].text
            elif f == "v":
                for i in range(len(t.cells)):
                    for j in range(len(t.cells[i])):
                        if t.cells[i][j].text.strip() == "":
                            if t.cells[i][j].vspan and not t.cells[i][j].top:
                                t.cells[i][j].text = t.cells[i - 1][j].text
        return t

    def _generate_image(self):
        from ..ext.ghostscript import Ghostscript

        self.imagename = "".join([self.rootname, ".png"])
        gs_call = "-q -sDEVICE=png16m -o {} -r300 {}".format(
            self.imagename, self.filename
        )
        gs_call = gs_call.encode().split()
        null = open(os.devnull, "wb")
        with Ghostscript(*gs_call, stdout=null) as gs:
            pass
        null.close()

    def _generate_table_bbox(self):
        def scale_areas(areas):
            scaled_areas = []
            for area in areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                x1, y1, x2, y2 = scale_pdf((x1, y1, x2, y2), image_scalers)
                scaled_areas.append((x1, y1, abs(x2 - x1), abs(y2 - y1)))
            return scaled_areas

        self.image, self.threshold = adaptive_threshold(
            self.imagename,
            process_background=self.process_background,
            blocksize=self.threshold_blocksize,
            c=self.threshold_constant,
        )

        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        pdf_width_scaler = self.pdf_width / float(image_width)
        pdf_height_scaler = self.pdf_height / float(image_height)
        image_scalers = (image_width_scaler, image_height_scaler, self.pdf_height)
        pdf_scalers = (pdf_width_scaler, pdf_height_scaler, image_height)

        if self.table_areas is None:
            regions = None
            if self.table_regions is not None:
                regions = scale_areas(self.table_regions)

            vertical_mask, vertical_segments = find_lines(
                self.threshold,
                regions=regions,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = find_lines(
                self.threshold,
                regions=regions,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            contours = find_contours(vertical_mask, horizontal_mask)
            table_bbox = find_joints(contours, vertical_mask, horizontal_mask)
        else:
            vertical_mask, vertical_segments = find_lines(
                self.threshold,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = find_lines(
                self.threshold,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            areas = scale_areas(self.table_areas)
            table_bbox = find_joints(areas, vertical_mask, horizontal_mask)

        self.table_bbox_unscaled = copy.deepcopy(table_bbox)

        self.table_bbox, self.vertical_segments, self.horizontal_segments = scale_image(
            table_bbox, vertical_segments, horizontal_segments, pdf_scalers
        )

    def _generate_columns_and_rows(self, table_idx, tk):
        # select elements which lie within table_bbox
        t_bbox = {}
        v_s, h_s = segments_in_bbox(
            tk, self.vertical_segments, self.horizontal_segments
        )
        t_bbox["horizontal"] = text_in_bbox(tk, self.horizontal_text)
        t_bbox["vertical"] = text_in_bbox(tk, self.vertical_text)

        t_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
        t_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))

        self.t_bbox = t_bbox

        cols, rows = zip(*self.table_bbox[tk])
        cols, rows = list(cols), list(rows)
        cols.extend([tk[0], tk[2]])
        rows.extend([tk[1], tk[3]])
        # sort horizontal and vertical segments
        cols = merge_close_lines(sorted(cols), line_tol=self.line_tol)
        rows = merge_close_lines(sorted(rows, reverse=True), line_tol=self.line_tol)
        # make grid using x and y coord of shortlisted rows and cols
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]

        return cols, rows, v_s, h_s

    def _generate_table(self, table_idx, cols, rows, **kwargs):
        v_s = kwargs.get("v_s")
        h_s = kwargs.get("h_s")
        if v_s is None or h_s is None:
            raise ValueError("No segments found on {}".format(self.rootname))

        table = Table(cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(v_s, h_s, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()
        # set spanning cells to True
        table = table.set_span()

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
                    indices = Lattice._reduce_index(
                        table, indices, shift_text=self.shift_text
                    )
                    for r_idx, c_idx, text in indices:
                        table.cells[r_idx][c_idx].text = text
        accuracy = compute_accuracy([[100, pos_errors]])

        if self.copy_text is not None:
            table = Lattice._copy_spanning_text(table, copy_text=self.copy_text)

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        whitespace = compute_whitespace(data)
        table.flavor = "lattice"
        table.accuracy = accuracy
        table.whitespace = whitespace
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table._image = (self.image, self.table_bbox_unscaled)
        table._segments = (self.vertical_segments, self.horizontal_segments)
        table._textedges = None

        return table

    def extract_tables(self, filename, suppress_stdout=False, layout_kwargs={}):
        self._generate_layout(filename, layout_kwargs)
        if not suppress_stdout:
            logger.info("Processing {}".format(os.path.basename(self.rootname)))

        if not self.horizontal_text:
            if self.images:
                warnings.warn(
                    "{} is image-based, camelot only works on"
                    " text-based pages.".format(os.path.basename(self.rootname))
                )
            else:
                warnings.warn(
                    "No tables found on {}".format(os.path.basename(self.rootname))
                )
            return []

        self._generate_image()
        self._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, tk in enumerate(
            sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            cols, rows, v_s, h_s = self._generate_columns_and_rows(table_idx, tk)
            table = self._generate_table(table_idx, cols, rows, v_s=v_s, h_s=h_s)
            table._bbox = tk
            _tables.append(table)

        return _tables
