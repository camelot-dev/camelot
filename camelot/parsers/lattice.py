# -*- coding: utf-8 -*-

import os

from .base import BaseParser
from ..utils import (
    build_file_path_in_temp_dir,
    export_pdf_as_png,
    scale_image,
    scale_pdf,
    segments_in_bbox,
    text_in_bbox_per_axis,
    merge_close_lines,
)
from ..image_processing import (
    adaptive_threshold,
    find_lines,
    find_contours,
    find_joints,
)


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

        For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_. # noqa
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
            shift_text=None,
            split_text=False,
            flag_size=False,
            strip_text="",
            line_tol=2,
            joint_tol=2,
            threshold_blocksize=15,
            threshold_constant=-2,
            iterations=0,
            resolution=300,
            **kwargs):
        super().__init__(
            "lattice",
            table_regions=table_regions,
            table_areas=table_areas,
            split_text=split_text,
            strip_text=strip_text,
            copy_text=copy_text,
            shift_text=shift_text or ["l", "t"],
            flag_size=flag_size,
        )
        self.process_background = process_background
        self.line_scale = line_scale
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.threshold_blocksize = threshold_blocksize
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution
        self.image_path = None
        self.pdf_image = None

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

    def record_parse_metadata(self, table):
        """Record data about the origin of the table
        """
        super().record_parse_metadata(table)
        # for plotting
        table._image = self.pdf_image  # Reuse the image used for calc
        table._segments = (self.vertical_segments, self.horizontal_segments)

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

        self.image_path = build_file_path_in_temp_dir(
            os.path.basename(self.filename),
            ".png"
        )
        export_pdf_as_png(self.filename, self.image_path, self.resolution)
        self.pdf_image, self.threshold = adaptive_threshold(
            self.image_path,
            process_background=self.process_background,
            blocksize=self.threshold_blocksize,
            c=self.threshold_constant,
        )

        image_width = self.pdf_image.shape[1]
        image_height = self.pdf_image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        pdf_width_scaler = self.pdf_width / float(image_width)
        pdf_height_scaler = self.pdf_height / float(image_height)
        image_scalers = (image_width_scaler,
                         image_height_scaler, self.pdf_height)
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

        [
            self.table_bbox_parses,
            self.vertical_segments,
            self.horizontal_segments
        ] = scale_image(
            table_bbox, vertical_segments, horizontal_segments, pdf_scalers
        )

        for bbox, parse in self.table_bbox_parses.items():
            joints = parse["joints"]

            # Merge x coordinates that are close together
            line_tol = self.line_tol
            # Sort the joints, make them a list of lists (instead of sets)
            joints_normalized = list(
                map(
                    lambda x: list(x),
                    sorted(joints, key=lambda j: - j[0])
                )
            )
            for idx in range(1, len(joints_normalized)):
                x_left, x_right = \
                    joints_normalized[idx-1][0], joints_normalized[idx][0]
                if x_left - line_tol <= x_right <= x_left + line_tol:
                    joints_normalized[idx][0] = x_left

            # Merge y coordinates that are close together
            joints_normalized = sorted(joints_normalized, key=lambda j: -j[1])
            for idx in range(1, len(joints_normalized)):
                y_bottom, y_top = \
                    joints_normalized[idx-1][1], joints_normalized[idx][1]
                if y_bottom - line_tol <= y_top <= y_bottom + line_tol:
                    joints_normalized[idx][1] = y_bottom

            # FRHTODO: check this is useful, otherwise get rid of the code
            # above
            parse["joints_normalized"] = joints_normalized

            cols = list(map(lambda coords: coords[0], joints))
            cols.extend([bbox[0], bbox[2]])
            rows = list(map(lambda coords: coords[1], joints))
            rows.extend([bbox[1], bbox[3]])

            # sort horizontal and vertical segments
            cols = merge_close_lines(sorted(cols), line_tol=self.line_tol)
            rows = merge_close_lines(
                sorted(rows, reverse=True),
                line_tol=self.line_tol
            )
            parse["col_anchors"] = cols
            parse["row_anchors"] = rows

    def _generate_columns_and_rows(self, bbox, user_cols):
        # select elements which lie within table_bbox
        v_s, h_s = segments_in_bbox(
            bbox, self.vertical_segments, self.horizontal_segments
        )
        self.t_bbox = text_in_bbox_per_axis(
            bbox,
            self.horizontal_text,
            self.vertical_text
            )
        parse = self.table_bbox_parses[bbox]

        # make grid using x and y coord of shortlisted rows and cols
        cols = [
            (parse["col_anchors"][i], parse["col_anchors"][i + 1])
            for i in range(0, len(parse["col_anchors"]) - 1)
        ]
        rows = [
            (parse["row_anchors"][i], parse["row_anchors"][i + 1])
            for i in range(0, len(parse["row_anchors"]) - 1)
        ]
        return cols, rows, v_s, h_s

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        v_s = kwargs.get("v_s")
        h_s = kwargs.get("h_s")
        if v_s is None or h_s is None:
            raise ValueError("No segments found on {}".format(self.rootname))

        table = self._initialize_new_table(table_idx, bbox, cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(v_s, h_s, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()
        # set spanning cells to True
        table = table.set_span()

        self.record_parse_metadata(table)
        return table
