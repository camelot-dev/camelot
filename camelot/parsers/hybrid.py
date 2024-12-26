"""Implementation of hybrid table parser."""

import numpy as np

from ..utils import bboxes_overlap
from ..utils import boundaries_to_split_lines
from .base import BaseParser
from .lattice import Lattice
from .network import Network


class Hybrid(BaseParser):
    """Defines a hybrid parser, leveraging both network and lattice parsers.

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
        edge_tol=None,
        row_tol=2,
        column_tol=0,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            "hybrid",
            table_regions=table_regions,
            table_areas=table_areas,
            flag_size=flag_size,
            split_text=split_text,
            strip_text=strip_text,
            debug=debug,
        )
        self.columns = columns  # Columns settings impacts the hybrid table
        self.network_parser = Network(
            table_regions=table_regions,
            table_areas=table_areas,
            columns=columns,
            flag_size=flag_size,
            split_text=split_text,
            strip_text=strip_text,
            edge_tol=edge_tol,
            row_tol=row_tol,
            column_tol=column_tol,
            debug=debug,
        )
        self.lattice_parser = Lattice(
            table_regions=table_regions,
            table_areas=table_areas,
            flag_size=flag_size,
            split_text=split_text,
            strip_text=strip_text,
            edge_tol=edge_tol,
            row_tol=row_tol,
            column_tol=column_tol,
            debug=debug,
        )

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
        """Call this method to prepare the page parsing .

        Parameters
        ----------
        filename : [type]
            [description]
        layout : [type]
            [description]
        dimensions : [type]
            [description]
        page_idx : [type]
            [description]
        layout_kwargs : [type]
            [description]
        """
        super().prepare_page_parse(
            filename,
            layout,
            dimensions,
            page_idx,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs,
        )
        self.network_parser.prepare_page_parse(
            filename,
            layout,
            dimensions,
            page_idx,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs,
        )
        self.lattice_parser.prepare_page_parse(
            filename,
            layout,
            dimensions,
            page_idx,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs,
        )

    def _generate_columns_and_rows(self, bbox, table_idx):
        parser = self.table_bbox_parses[bbox]
        return parser._generate_columns_and_rows(bbox, table_idx)

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        parser = self.table_bbox_parses[bbox]
        table = parser._generate_table(table_idx, bbox, cols, rows, **kwargs)
        # Because hybrid can inject extraneous splits from both lattice and
        # network, remove lines / cols that are completely empty.
        table.df = table.df.replace("", np.nan)
        table.df = table.df.dropna(axis=0, how="all")
        table.df = table.df.dropna(axis=1, how="all")
        table.df = table.df.replace(np.nan, "")
        table.shape = table.df.shape
        return table

    @staticmethod
    def _augment_boundaries_with_splits(boundaries, splits, tolerance=0):
        """Augment existing boundaries using provided hard splits.

        Boundaries:   |---|    |-| |---------|  #noqa RST305
        Splits:     |       |     |       |  #noqa RST305
        Augmented:  |-------|-----|-------|--|  #noqa RST305
        """
        idx_boundaries = len(boundaries) - 1
        idx_splits = len(splits) - 1
        previous_boundary = None
        while True:
            if idx_splits < 0:
                # No more splits to incorporate, we're done
                break
            split = splits[idx_splits]

            if idx_boundaries < 0:
                # Need to insert remaining splits
                new_boundary = [split, boundaries[0][0]]
                boundaries.insert(0, new_boundary)
                idx_splits = idx_splits - 1
            else:
                boundary = boundaries[idx_boundaries]
                if boundary[1] < split + tolerance:
                    # The lattice column is further to the right of our
                    # col boundary.  We move our left boundary to match.
                    boundary[1] = split
                    # And if there was another segment after, we make its
                    # right boundary match as well so that there's no gap
                    if previous_boundary is not None:
                        previous_boundary[0] = split
                    idx_splits = idx_splits - 1
                elif boundary[0] > split - tolerance:
                    # Our boundary is fully after the split, move on
                    idx_boundaries = idx_boundaries - 1
                    previous_boundary = boundary
                    if idx_boundaries < 0:
                        # If this is the last boundary to the left, set its
                        # edge at the split
                        boundary[0] = split
                        idx_splits = idx_splits - 1
                else:
                    # The split is inside our boundary: split it
                    new_boundary = [split, boundary[1]]
                    boundaries.insert(idx_boundaries + 1, new_boundary)
                    boundary[1] = split
                    previous_boundary = new_boundary
                    idx_splits = idx_splits - 1
        return boundaries

    def _merge_bbox_analysis(self, lattice_bbox, network_bbox):
        """Identify splits that were only detected by lattice or by network."""
        lattice_parse = self.lattice_parser.table_bbox_parses[lattice_bbox]
        lattice_cols = lattice_parse["col_anchors"]

        network_bbox_data = self.network_parser.table_bbox_parses[network_bbox]
        network_cols_boundaries = network_bbox_data["cols_boundaries"]

        # Favor network, but complete or adjust its columns based on the
        # splits identified by lattice.
        if network_cols_boundaries is None:
            self.table_bbox_parses[lattice_bbox] = self.lattice_parser
        else:
            network_cols_boundaries = self._augment_boundaries_with_splits(
                network_cols_boundaries, lattice_cols, self.lattice_parser.joint_tol
            )
            augmented_bbox = (
                network_cols_boundaries[0][0],
                min(lattice_bbox[1], network_bbox[1]),
                network_cols_boundaries[-1][1],
                max(lattice_bbox[3], network_bbox[3]),
            )
            network_bbox_data["cols_anchors"] = boundaries_to_split_lines(
                network_cols_boundaries
            )

            del self.network_parser.table_bbox_parses[network_bbox]
            self.network_parser.table_bbox_parses[augmented_bbox] = network_bbox_data
            self.table_bbox_parses[augmented_bbox] = self.network_parser

    def _generate_table_bbox(self):
        # Collect bboxes from both parsers
        self.lattice_parser._generate_table_bbox()
        _lattice_bboxes = sorted(
            self.lattice_parser.table_bbox_parses, key=lambda bbox: (bbox[0], -bbox[1])
        )
        self.network_parser._generate_table_bbox()
        _network_bboxes = sorted(
            self.network_parser.table_bbox_parses, key=lambda bbox: (bbox[0], -bbox[1])
        )

        # Merge the data from both processes
        for lattice_bbox in _lattice_bboxes:
            merged = False

            for idx in range(len(_network_bboxes) - 1, -1, -1):
                network_bbox = _network_bboxes[idx]
                if not bboxes_overlap(lattice_bbox, network_bbox):
                    continue
                self._merge_bbox_analysis(lattice_bbox, network_bbox)
                # network_bbox_data["cols_boundaries"]
                del _network_bboxes[idx]
                merged = True
            if not merged:
                self.table_bbox_parses[lattice_bbox] = self.lattice_parser

        # Add the bboxes from network that haven't been merged
        for network_bbox in _network_bboxes:
            self.table_bbox_parses[network_bbox] = self.network_parser
