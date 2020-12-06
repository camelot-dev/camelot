# -*- coding: utf-8 -*-

import os
import copy
import logging
import subprocess

try:
    import easyocr
except ImportError:
    _HAS_EASYOCR = False
else:
    _HAS_EASYOCR = True

import pandas as pd
from PIL import Image

from .base import BaseParser
from ..core import Table
from ..utils import TemporaryDirectory, merge_close_lines, scale_pdf, segments_in_bbox
from ..image_processing import (
    adaptive_threshold,
    find_lines,
    find_contours,
    find_joints,
)


logger = logging.getLogger("camelot")


class LatticeOCR(BaseParser):
    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        line_scale=15,
        line_tol=2,
        joint_tol=2,
        threshold_blocksize=15,
        threshold_constant=-2,
        iterations=0,
        resolution=300,
    ):
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.line_scale = line_scale
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.threshold_blocksize = threshold_blocksize
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution

        if _HAS_EASYOCR:
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            raise ImportError("easyocr is required to run OCR on image-based PDFs.")

    def _generate_image(self):
        from ..ext.ghostscript import Ghostscript

        self.imagename = "".join([self.rootname, ".png"])
        gs_call = "-q -sDEVICE=png16m -o {} -r900 {}".format(
            self.imagename, self.filename
        )
        gs_call = gs_call.encode().split()
        null = open(os.devnull, "wb")
        with Ghostscript(*gs_call, stdout=null) as gs:
            pass
        null.close()

    def _generate_table_bbox(self):
        def scale_areas(areas, scalers):
            scaled_areas = []
            for area in areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                x1, y1, x2, y2 = scale_pdf((x1, y1, x2, y2), scalers)
                scaled_areas.append((x1, y1, abs(x2 - x1), abs(y2 - y1)))
            return scaled_areas

        self.image, self.threshold = adaptive_threshold(
            self.imagename, blocksize=self.threshold_blocksize, c=self.threshold_constant
        )

        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        image_scalers = (image_width_scaler, image_height_scaler, self.pdf_height)

        if self.table_areas is None:
            regions = None
            if self.table_regions is not None:
                regions = scale_areas(self.table_regions, image_scalers)

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

            areas = scale_areas(self.table_areas, image_scalers)
            table_bbox = find_joints(areas, vertical_mask, horizontal_mask)

        self.table_bbox_unscaled = copy.deepcopy(table_bbox)

        self.table_bbox = table_bbox
        self.vertical_segments = vertical_segments
        self.horizontal_segments = horizontal_segments

    def _generate_columns_and_rows(self, table_idx, tk):
        cols, rows = zip(*self.table_bbox[tk])
        cols, rows = list(cols), list(rows)
        cols.extend([tk[0], tk[2]])
        rows.extend([tk[1], tk[3]])
        # sort horizontal and vertical segments
        cols = merge_close_lines(sorted(cols), line_tol=self.line_tol)
        rows = merge_close_lines(sorted(rows), line_tol=self.line_tol)
        # make grid using x and y coord of shortlisted rows and cols
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]

        return cols, rows


    def _generate_table(self, table_idx, cols, rows, **kwargs):
        table = Table(cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(self.vertical_segments, self.horizontal_segments, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()
        # set spanning cells to True
        table = table.set_span()

        _seen = set()
        for r_idx in range(len(table.cells)):
            for c_idx in range(len(table.cells[r_idx])):
                if (r_idx, c_idx) in _seen:
                    continue

                _seen.add((r_idx, c_idx))

                _r_idx = r_idx
                _c_idx = c_idx

                if table.cells[r_idx][_c_idx].hspan:
                    while not table.cells[r_idx][_c_idx].right:
                        _c_idx += 1
                        _seen.add((r_idx, _c_idx))

                if table.cells[_r_idx][c_idx].vspan:
                    while not table.cells[_r_idx][c_idx].bottom:
                        _r_idx += 1
                        _seen.add((_r_idx, c_idx))

                for i in range(r_idx, _r_idx + 1):
                    for j in range(c_idx, _c_idx + 1):
                        _seen.add((i, j))

                x1 = int(table.cells[r_idx][c_idx].x1)
                y1 = int(table.cells[_r_idx][_c_idx].y1)

                x2 = int(table.cells[_r_idx][_c_idx].x2)
                y2 = int(table.cells[r_idx][c_idx].y2)

                with TemporaryDirectory() as tempdir:
                    temp_image_path = os.path.join(tempdir, f"{table_idx}_{r_idx}_{c_idx}.png")

                    cell_image = Image.fromarray(self.image[y2:y1, x1:x2])
                    cell_image.save(temp_image_path)

                    text = self.reader.readtext(temp_image_path, detail=0)
                    text = " ".join(text)

                table.cells[r_idx][c_idx].text = text

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        table.flavor = "lattice_ocr"
        table.accuracy = 0
        table.whitespace = 0
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        table._text = None
        table._image = (self.image, self.table_bbox_unscaled)
        table._segments = (self.vertical_segments, self.horizontal_segments)
        table._textedges = None

        return table

    def extract_tables(self, filename, suppress_stdout=False, layout_kwargs={}):
        self._generate_layout(filename, layout_kwargs)
        if not suppress_stdout:
            logger.info("Processing {}".format(os.path.basename(self.rootname)))

        self._generate_image()
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
