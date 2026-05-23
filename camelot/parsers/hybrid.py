"""Implementation of hybrid table parser."""

from ..utils import bboxes_overlap
from ..utils import boundaries_to_split_lines
from ..utils import text_in_bbox
from .base import BaseParser
from .lattice import Lattice
from .network import Network

#: Minimum fraction of a lattice grid's (cols x rows) crossing points that
#: must carry an actual joint for the grid to count as "complete" — a real
#: lattice of ruled lines rather than a couple of stray rules. Below this,
#: hybrid keeps augmenting lattice's boundaries with network's text splits.
_LATTICE_GRID_COVERAGE = 0.5

#: A complete grid's row count must stay *commensurate* with the
#: column-aligned text rows inside it: at least ``_LATTICE_ROW_MATCH`` of them
#: (else lattice is a partially-ruled fragment dropping unruled rows — the
#: us-008 / us-033 failure mode) and at most ``_LATTICE_ROW_CEIL`` times them
#: (else lattice's interior rules don't separate real multi-column rows —
#: spurious rules or a complex multi-level header that network handles
#: better, e.g. the vertical_header fixture). Outside the band, hybrid keeps
#: the network-augmented path. See :meth:`Hybrid._count_column_aligned_rows`.
_LATTICE_ROW_MATCH = 0.55
_LATTICE_ROW_CEIL = 1.5

#: The row-match band is only trusted when a grid has at least this many
#: column-aligned rows — below it the count is too small to be meaningful
#: (e.g. a list-like ruled table with one multi-column row), so the band is
#: skipped and the grid kept on its joint-coverage merit alone.
_MIN_ALIGNED_ROWS = 3


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
    strip_text : str or sequence of str, optional (default: '')
        Characters or substrings to strip from each cell. A ``str``
        strips per-character; a list/tuple of ``str`` strips whole
        substrings (#484).
    edge_tol : int, optional (default: 50)
        Tolerance parameter for extending textedges vertically.
    row_tol : int, optional (default: 2)
        Tolerance parameter used to combine text vertically,
        to generate rows.
    column_tol : int, optional (default: 0)
        Tolerance parameter used to combine text horizontally,
        to generate columns.
    engine : str, optional (default: 'raster')
        Line-detection engine for hybrid's **lattice half** (the network
        half is text-based and unaffected):

        - ``'raster'`` (default): detect ruled lines with OpenCV on the
          rendered page.
        - ``'combined'``: OpenCV **plus** the PDF's native vector ruled
          lines unioned in — recovers faintly-rendered rules.
        - ``'vector'``: detect ruled lines **straight from the PDF's vector
          graphics, skipping rasterisation and OpenCV entirely** — the
          render-free hybrid (network text-edge alignment merged with vector
          ruled lines) for partial-ruled / borderless tables at roughly an
          order of magnitude less time than the raster path. (#39)

    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        columns=None,
        flag_size=False,
        split_text=False,
        strip_text="",
        replace_text=None,
        edge_tol=None,
        row_tol=2,
        column_tol=0,
        debug=False,
        engine="raster",
        **kwargs,
    ):
        super().__init__(
            "hybrid",
            table_regions=table_regions,
            table_areas=table_areas,
            flag_size=flag_size,
            split_text=split_text,
            strip_text=strip_text,
            replace_text=replace_text,
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
            replace_text=replace_text,
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
            replace_text=replace_text,
            edge_tol=edge_tol,
            row_tol=row_tol,
            column_tol=column_tol,
            debug=debug,
            # Forward the line-detection engine so flavor='hybrid' can drive
            # its lattice half with 'raster', 'combined' (raster + PDF vector
            # lines, #763) or the render-free 'vector' engine (#39).
            engine=engine,
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
        rotation,
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
            rotation,
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
            rotation,
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
            rotation,
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
        # drop empty rows
        table.df = table.df.loc[~(table.df == "").all(axis=1)]
        # drop empty columns
        table.df = table.df.loc[:, ~(table.df == "").all(axis=0)]
        table.shape = table.df.shape
        return table

    def _reject_table(self, table) -> bool:
        """Drop tables left empty after the empty-row/col purge.

        The render-free ``engine='vector'`` half reads ruled lines straight
        from the PDF's vector graphics, which include decorative page borders
        and form rules. Those can raise a "grid" with no text inside; once
        :meth:`_generate_table` strips its all-empty rows and columns nothing
        is left, and an empty table would otherwise leak out as a spurious
        detection. (The rendered raster/combined halves don't hit this — the
        OpenCV pipeline doesn't pick those rules up — so their output is
        unchanged.)
        """
        return table.df.empty or table.shape[0] == 0 or table.shape[1] == 0

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

    def _count_column_aligned_rows(self, lattice_bbox, col_anchors):
        """Count text rows inside ``lattice_bbox`` that populate >=2 columns.

        Used to tell a complete ruled grid from a *partially*-ruled fragment.
        A horizontal rule lattice missed leaves text in **several** columns at
        the same y; a multi-line cell only adds extra text in **one** column.
        So clustering the bbox's textlines by y and counting the clusters that
        span at least two of lattice's columns approximates the table's true
        row count — robust to multi-line cells (which inflate a naive y-count).
        """
        tls = [
            t
            for t in text_in_bbox(
                lattice_bbox, self.horizontal_text + self.vertical_text
            )
            if t.get_text().strip()
        ]
        if not tls:
            return 0

        def column_of(textline):
            xc = (textline.x0 + textline.x1) / 2.0
            for i in range(len(col_anchors) - 1):
                if col_anchors[i] <= xc <= col_anchors[i + 1]:
                    return i
            return None

        rows = self.network_parser._group_rows(tls, row_tol=self.network_parser.row_tol)
        aligned = 0
        for row in rows:
            cols = {column_of(t) for t in row}
            cols.discard(None)
            if len(cols) >= 2:
                aligned += 1
        return aligned

    def _lattice_grid_is_complete(self, lattice_bbox):
        """Whether lattice already resolved a full ruled grid for this bbox.

        The combine ``_augment_boundaries_with_splits`` *unions* network's
        text-derived column splits onto lattice's. On a partial / borderless
        table that recovers columns lattice couldn't see — the niche hybrid
        is for. But on an **already-complete** ruled grid the union only adds
        spurious splits and, because the merged bbox is then parsed by the
        *network* parser (text-grouped rows), it also throws away lattice's
        exact ruled rows — the over-split that sinks fully-ruled docs (#38).

        So gate the augmentation. A grid counts as complete only when:

        1. lattice found genuine ruled lines in **both** directions (interior
           column *and* row anchors, not just the two bbox edges);
        2. its joints actually cover that grid (a real lattice of crossings,
           not a couple of stray rules); and
        3. lattice's row lines account for the table's text rows — i.e. it is
           not a *partially*-ruled fragment whose unruled rows lattice would
           silently drop (the us-008 / us-033 failure mode). Network handles
           those better, so they stay on the augmented path.

        Complete grids are routed to the lattice parser as-is; incomplete ones
        keep the network-augmented path.
        """
        parse = self.lattice_parser.table_bbox_parses[lattice_bbox]
        col_anchors = parse["col_anchors"]
        row_anchors = parse["row_anchors"]
        # Need at least one interior line in *each* direction (more than the
        # two bbox edges) — otherwise lattice has no grid of its own to keep.
        if len(col_anchors) <= 2 or len(row_anchors) <= 2:
            return False
        joints_normalized = parse.get("joints_normalized", [])
        unique_joints = {(round(j[0], 1), round(j[1], 1)) for j in joints_normalized}
        grid_points = len(col_anchors) * len(row_anchors)
        coverage = len(unique_joints) / grid_points if grid_points else 0.0
        if coverage < _LATTICE_GRID_COVERAGE:
            return False
        # Lattice's row count must stay commensurate with the column-aligned
        # text rows: too few => partially-ruled fragment (us-008); too many
        # => interior rules that don't separate real rows (vertical headers).
        lattice_rows = len(row_anchors) - 1
        aligned_rows = self._count_column_aligned_rows(lattice_bbox, col_anchors)
        if aligned_rows >= _MIN_ALIGNED_ROWS and not (
            _LATTICE_ROW_MATCH * aligned_rows
            <= lattice_rows
            <= _LATTICE_ROW_CEIL * aligned_rows
        ):
            return False
        return True

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
        elif self._lattice_grid_is_complete(lattice_bbox):
            # Lattice already has a full ruled grid here — keep it intact
            # instead of unioning network's splits on top (#38 over-split).
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
        self.table_bbox_parses = {}
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
