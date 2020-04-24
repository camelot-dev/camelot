# -*- coding: utf-8 -*-
"""Implementation of hybrid table parser."""

from __future__ import division

import numpy as np
import copy
import warnings

from .base import BaseParser
from ..core import (TextAlignment, TextAlignments, ALL_ALIGNMENTS)
from ..utils import (
    get_index_closest_point,
    get_textline_coords,
    bbox_from_str,
    text_in_bbox,
    text_in_bbox_per_axis,
    bbox_from_text,
    distance_tl_to_bbox,
    find_columns_coordinates
)

from matplotlib import patches as patches

# maximum number of columns over which a header can spread
MAX_COL_SPREAD_IN_HEADER = 3


def search_header_from_body_bbox(body_bbox, textlines, col_anchors, max_v_gap):
    """Expand a bbox vertically up by looking for plausible headers.

    The core algorithm is based on fairly strict alignment of text. It works
    for the table body, but might fail on tables' headers since they tend to be
    in a different font, alignment (e.g. vertical), etc.
    This method evalutes the area above the table body's bbox for
    characteristics of a table header: close to the top of the body, with cells
    that fit within the horizontal bounds identified.
    """
    new_bbox = body_bbox
    (left, bottom, right, top) = body_bbox
    zones = []

    def column_spread(left, right, col_anchors):
        """Get the number of columns crossed by a segment [left, right]."""
        indexLeft = 0
        while indexLeft < len(col_anchors) \
                and col_anchors[indexLeft] < left:
            indexLeft += 1
        indexRight = indexLeft
        while indexRight < len(col_anchors) \
                and col_anchors[indexRight] < right:
            indexRight += 1

        return indexRight - indexLeft

    keep_searching = True
    while keep_searching:
        keep_searching = False
        # a/ first look for the closest text element above the bbox.
        # It will be the anchor for a possible new row.
        closest_above = None
        all_above = []
        for te in textlines:
            # higher than the table, >50% within its bounds
            te_center = 0.5 * (te.x0 + te.x1)
            if te.y0 > top and left < te_center < right:
                all_above.append(te)
                if closest_above is None or closest_above.y0 > te.y0:
                    closest_above = te

        if closest_above and \
                closest_above.y0 < top + max_v_gap:
            # b/ We have a candidate cell that is within the correct
            # vertical band, and directly above the table. Starting from
            # this anchor, we list all the textlines within the same row.
            tls_in_new_row = []
            top = closest_above.y1
            pushed_up = True
            while pushed_up:
                pushed_up = False
                # Iterate and extract elements that fit in the row
                # from our list
                for i in range(len(all_above) - 1, -1, -1):
                    te = all_above[i]
                    if te.y0 < top:
                        # The bottom of this element is within our row
                        # so we add it.
                        tls_in_new_row.append(te)
                        all_above.pop(i)
                        if te.y1 > top:
                            # If the top of this element raises our row's
                            # band, we'll need to keep on searching for
                            # overlapping items
                            top = te.y1
                            pushed_up = True

            # Get the x-ranges for all the textlines, and merge the
            # x-ranges that overlap
            zones = zones + \
                list(map(lambda tl: [tl.x0, tl.x1], tls_in_new_row))
            zones.sort(key=lambda z: z[0])  # Sort by left coordinate
            # Starting from the right, if two zones overlap horizontally,
            # merge them
            merged_something = True
            while merged_something:
                merged_something = False
                for i in range(len(zones) - 1, 0, -1):
                    zone_right = zones[i]
                    zone_left = zones[i-1]
                    if zone_left[1] >= zone_right[0]:
                        zone_left[1] = max(zone_right[1], zone_left[1])
                        zones.pop(i)
                        merged_something = True

            max_spread = max(
                list(
                    map(
                        lambda zone: column_spread(
                            zone[0], zone[1], col_anchors),
                        zones
                    )
                )
            )
            if max_spread <= MAX_COL_SPREAD_IN_HEADER:
                # Combined, the elements we've identified don't cross more
                # than the authorized number of columns.
                # We're trying to avoid
                # 0: <BAD: Added header spans too broad>
                # 1: <A1>    <B1>    <C1>    <D1>    <E1>
                # 2: <A2>    <B2>    <C2>    <D2>    <E2>
                # if len(zones) > TEXTEDGE_REQUIRED_ELEMENTS:
                new_bbox = (left, bottom, right, top)

                # At this stage we've identified a plausible row (or the
                # beginning of one).
                keep_searching = True
    return new_bbox


class Alignments(object):
    """
    Represent the number of textlines aligned with this one across each edge.

    A cell can be vertically aligned with others by having matching left,
    right, or middle edge, and horizontally aligned by having matching top,
    bottom, or center edge.

    """

    def __init__(self):
        # Vertical alignments
        self.left = 0
        self.right = 0
        self.middle = 0

        # Horizontal alignments
        self.bottom = 0
        self.top = 0
        self.center = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def max_v(self):
        """Returns the maximum number of alignments along
        one of the vertical axis (left/right/middle).
        """
        return max(self.left, self.right, self.middle)

    def max_h(self):
        """Returns the maximum number of alignments along
        one of the horizontal axis (bottom/top/center).
        """
        return max(self.bottom, self.top, self.center)

    def max_v_edge_name(self):
        """Returns the name of the vertical edge that has the
        maximum number of alignments.
        """
        return max(
            ["left", "right", "middle"],
            key=lambda edge_name: self[edge_name]
        )

    def max_h_edge_name(self):
        """Returns the name of the horizontal edge that has the
        maximum number of alignments.
        """
        return max(
            ["bottom", "top", "center"],
            key=lambda edge_name: self[edge_name]
        )

    def alignment_score(self):
        """We define the alignment score of a textline as the product of the
        number of aligned elements - 1. The -1 is to avoid favoring
         singletons on a long line.
        """
        return (self.max_v()-1) * (self.max_h()-1)


class TextEdges2(TextAlignments):
    """Defines a dict of vertical (top, bottom, middle) and
    horizontal (left, right, and middle) text alignments found on
    the PDF page. The dict has three keys based on the alignments,
    and each key's value is a list of camelot.core.TextEdge objects.
    """

    def __init__(self):
        super().__init__(ALL_ALIGNMENTS)
        # For each textline, dictionary "edge type" to
        # "number of textlines aligned"
        self._textlines_alignments = {}

        # Maximum number of distinct aligned elements in rows/cols
        self.max_rows = None
        self.max_cols = None

    @staticmethod
    def _create_new_text_edge(coord, textline, align):
        return TextAlignment(coord, textline, align)

    def _update_edge(self, edge, coord, textline):
        edge.register_aligned_textline(textline, coord)

    def _register_all_text_lines(self, textlines):
        """Add all textlines to our edge repository to
        identify alignments.
        """
        # Identify all the edge alignments
        for tl in textlines:
            if len(tl.get_text().strip()) > 0:
                self._register_textline(tl)

    def _compute_alignment_counts(self):
        """Build a dictionary textline -> alignment object.
        """
        for edge_name, textedges in self._textedges.items():
            for textedge in textedges:
                for textline in textedge.textlines:
                    alignments = self._textlines_alignments.get(
                        textline, None)
                    if alignments is None:
                        alignments = Alignments()
                        self._textlines_alignments[textline] = alignments
                    alignments[edge_name] = len(textedge.textlines)

        # Finally calculate the overall maximum number of rows/cols
        self.max_rows = max(
            map(
                lambda alignments: alignments.max_h(),
                self._textlines_alignments.values()
            ),
            default=0
        )
        self.max_cols = max(
            map(
                lambda alignments: alignments.max_v(),
                self._textlines_alignments.values()
            ),
            default=0
        )

    def _calculate_gaps_thresholds(self, percentile=75):
        """Identify reasonable gaps between lines and columns based
        on gaps observed across alignments.
        This can be used to reject cells as too far away from
        the core table.
        """
        h_gaps, v_gaps = [], []
        for edge_name in self._textedges:
            edge_array = self._textedges[edge_name]
            gaps = []
            vertical = edge_name in ["left", "right", "middle"]
            sort_function = (lambda tl: tl.y0) \
                if vertical \
                else (lambda tl: tl.x0)
            for alignments in edge_array:
                tls = sorted(
                    alignments.textlines,
                    key=sort_function,
                    reverse=True
                )
                for i in range(1, len(tls)):
                    # If the lines are vertically aligned (stacked up), we
                    # record the vertical gap between them
                    if vertical:
                        gap = tls[i-1].y1 - tls[i].y0
                    else:
                        gap = tls[i-1].x1 - tls[i].x0
                    gaps.append(gap)
            if gaps:
                if vertical:
                    v_gaps.append(np.percentile(gaps, percentile))
                else:
                    h_gaps.append(np.percentile(gaps, percentile))
                direction_str = 'vertical' if vertical else 'horizontal'
                rounded_gaps = list(map(lambda x: round(x, 2), gaps))
                print(
                    f"{direction_str} gaps found "
                    f"for {edge_name}: "
                    f"{rounded_gaps} "
                    f"with {percentile}th percentile "
                    f"{np.percentile(gaps, percentile)}"
                )
        return max(h_gaps, default=None), max(v_gaps, default=None)

    def _remove_unconnected_edges(self):
        """Weed out elements which are only connected to others vertically
        or horizontally. There needs to be connections across both
        dimensions.
        """
        removed_singletons = True
        while removed_singletons:
            removed_singletons = False
            for edge_type in self._textedges:
                # For each alignment edge, remove items if they are singletons
                # either horizontally or vertically
                for te in self._textedges[edge_type]:
                    for i in range(len(te.textlines) - 1, -1, -1):
                        tl = te.textlines[i]
                        alignments = self._textlines_alignments[tl]
                        if alignments.max_h() <= 1 or alignments.max_v() <= 1:
                            del te.textlines[i]
                            removed_singletons = True
            self._textlines_alignments = {}
            self._compute_alignment_counts()

    def _most_connected_textline(self):
        """ Retrieve the textline that is most connected across vertical and
        horizontal axis.

        """
        # Find the textline with the highest alignment score
        return max(
            self._textlines_alignments.keys(),
            key=lambda textline:
                self._textlines_alignments[textline].alignment_score(),
            default=None
        )

    def _compute_plausible_gaps(self):
        """ Evaluate plausible gaps between cells horizontally and vertically
        based on the textlines aligned with the most connected textline.

        Returns
        -------
        gaps_hv : tuple
            (horizontal_gap, horizontal_gap) in pdf coordinate space.

        """
        if self.max_rows <= 1 or self.max_cols <= 1:
            return None

        # Determine the textline that has the most combined
        # alignments across horizontal and vertical axis.
        # It will serve as a reference axis along which to collect the average
        # spacing between rows/cols.
        most_aligned_tl = self._most_connected_textline()
        most_aligned_coords = get_textline_coords(
            most_aligned_tl)

        # Retrieve the list of textlines it's aligned with, across both
        # axis
        best_alignment = self._textlines_alignments[most_aligned_tl]
        ref_h_edge_name = best_alignment.max_h_edge_name()
        ref_v_edge_name = best_alignment.max_v_edge_name()
        best_h_textedges = self._textedges[ref_h_edge_name]
        best_v_textedges = self._textedges[ref_v_edge_name]
        h_coord = most_aligned_coords[ref_h_edge_name]
        v_coord = most_aligned_coords[ref_v_edge_name]
        h_textlines = sorted(
            best_h_textedges[
                get_index_closest_point(
                    h_coord,
                    best_h_textedges,
                    fn=lambda x: x.coord
                )
            ].textlines,
            key=lambda tl: tl.x0,
            reverse=True
        )
        v_textlines = sorted(
            best_v_textedges[
                get_index_closest_point(
                    v_coord,
                    best_v_textedges,
                    fn=lambda x: x.coord
                )
            ].textlines,
            key=lambda tl: tl.y0,
            reverse=True
        )

        h_gaps, v_gaps = [], []
        for i in range(1, len(v_textlines)):
            v_gaps.append(v_textlines[i-1].y0 - v_textlines[i].y0)
        for i in range(1, len(h_textlines)):
            h_gaps.append(h_textlines[i-1].x0 - h_textlines[i].x0)

        if (not h_gaps or not v_gaps):
            return None
        percentile = 75
        gaps_hv = (
            2.0 * np.percentile(h_gaps, percentile),
            2.0 * np.percentile(v_gaps, percentile)
        )
        return gaps_hv

    def _build_bbox_candidate(self, gaps_hv, debug_info=None):
        """ Seed the process with the textline with the highest alignment
        score, then expand the bbox with textlines within threshold.

        Parameters
        ----------
        gaps_hv : tuple
             The maximum distance allowed to consider surrounding lines/columns
             as part of the same table.
        debug_info : array (optional)
            Optional parameter array, in which to store extra information
            to help later visualization of the table creation.
        """
        # First, determine the textline that has the most combined
        # alignments across horizontal and vertical axis.
        # It will serve both as a starting point for the table boundary
        # search, and as a way to estimate the average spacing between
        # rows/cols.
        most_aligned_tl = self._most_connected_textline()

        # Calculate the 75th percentile of the horizontal/vertical
        # gaps between textlines.  Use this as a reference for a threshold
        # to not exceed while looking for table boundaries.
        max_h_gap, max_v_gap = gaps_hv[0], gaps_hv[1]

        if debug_info is not None:
            # Store debug info
            debug_info_search = {
                "max_h_gap": max_h_gap,
                "max_v_gap": max_v_gap,
                "iterations": []
            }
            debug_info.append(debug_info_search)
        else:
            debug_info_search = None

        MINIMUM_TEXTLINES_IN_TABLE = 6
        bbox = (most_aligned_tl.x0, most_aligned_tl.y0,
                most_aligned_tl.x1, most_aligned_tl.y1)

        # For the body of the table, we only consider cells with alignments
        # on both axis.
        tls_search_space = list(self._textlines_alignments.keys())
        # tls_search_space = []
        tls_search_space.remove(most_aligned_tl)
        tls_in_bbox = [most_aligned_tl]
        last_bbox = None
        while last_bbox != bbox:
            if debug_info_search is not None:
                # Store debug info
                debug_info_search["iterations"].append(bbox)

            last_bbox = bbox
            # Go through all remaining textlines, expand our bbox
            # if a textline is within our proximity tolerance
            for i in range(len(tls_search_space) - 1, -1, -1):
                tl = tls_search_space[i]
                h_distance, v_distance = distance_tl_to_bbox(tl, bbox)

                # Move textline to our bbox and expand the bbox accordingly
                # if the textline is close.
                if h_distance < max_h_gap and v_distance < max_v_gap:
                    tls_in_bbox.append(tl)
                    bbox = (
                        min(bbox[0], tl.x0),
                        min(bbox[1], tl.y0),
                        max(bbox[2], tl.x1),
                        max(bbox[3], tl.y1)
                    )
                    del tls_search_space[i]
        if len(tls_in_bbox) > MINIMUM_TEXTLINES_IN_TABLE:
            return bbox
        else:
            print(f"Only {len(tls_in_bbox)}, that's not enough.")
            return None

    def generate(self, textlines):
        """Generate the text edge dictionaries based on the
        input textlines.
        """
        self._register_all_text_lines(textlines)
        self._compute_alignment_counts()

    def plot_alignments(self, ax):
        """Displays a visualization of the alignments as currently computed.
        """
        # FRHTODO: This is too busy and doesn't plot lines
        most_aligned_tl = sorted(
            self._textlines_alignments.keys(),
            key=lambda textline:
            self._textlines_alignments[textline].alignment_score(),
            reverse=True
        )[0]

        ax.add_patch(
            patches.Rectangle(
                (most_aligned_tl.x0, most_aligned_tl.y0),
                most_aligned_tl.x1 - most_aligned_tl.x0,
                most_aligned_tl.y1 - most_aligned_tl.y0,
                color="red",
                alpha=0.5
            )
        )
        for tl, alignments in self._textlines_alignments.items():
            ax.text(
                tl.x0 - 5,
                tl.y0 - 5,
                f"{alignments.max_h()}x{alignments.max_v()}",
                fontsize=5,
                color="black"
            )


class Hybrid(BaseParser):
    """Hybrid method of parsing looks for spaces between text
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
        flag_size=False,
        split_text=False,
        strip_text="",
        edge_tol=None,
        row_tol=2,
        column_tol=0,
        debug=False,
        **kwargs
    ):
        super().__init__(
            "hybrid",
            table_regions=table_regions,
            table_areas=table_areas,
            split_text=split_text,
            strip_text=strip_text,
            flag_size=flag_size,
            debug=debug
        )
        self.columns = columns
        self.textedges = None

        self._validate_columns()
        self.edge_tol = edge_tol
        self.row_tol = row_tol
        self.column_tol = column_tol

    # FRHTODO: Check if needed, refactor with Stream
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
        xmin = min(t.x0 for direction in t_bbox for t in t_bbox[direction])
        ymin = min(t.y0 for direction in t_bbox for t in t_bbox[direction])
        xmax = max(t.x1 for direction in t_bbox for t in t_bbox[direction])
        ymax = max(t.y1 for direction in t_bbox for t in t_bbox[direction])
        text_bbox = (xmin, ymin, xmax, ymax)
        return text_bbox

    # FRHTODO: Check if needed, refactor with Stream
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
        row_y = None
        rows = []
        temp = []
        non_empty_text = [t for t in text if t.get_text().strip()]
        for t in non_empty_text:
            # is checking for upright necessary?
            # if t.get_text().strip() and all([obj.upright \
            #   for obj in t._objs
            # if type(obj) is LTChar]):
            if row_y is None:
                row_y = t.y0
            elif not np.isclose(row_y, t.y0, atol=row_tol):
                rows.append(sorted(temp, key=lambda t: t.x0))
                temp = []
                # We update the row's bottom as we go, to be forgiving if there
                # is a gradual change across multiple columns.
                row_y = t.y0
            temp.append(t)
        rows.append(sorted(temp, key=lambda t: t.x0))
        return rows

    # FRHTODO: Check if needed, refactor with Stream
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
                        if np.isclose(higher[0], lower[1],
                                      atol=abs(column_tol)):
                            merged.append(higher)
                        else:
                            upper_bound = max(lower[1], higher[1])
                            lower_bound = min(lower[0], higher[0])
                            merged[-1] = (lower_bound, upper_bound)
                    else:
                        merged.append(higher)
        return merged

    # FRHTODO: Check if needed, refactor with Stream
    @staticmethod
    def _join_rows(rows_grouped, text_y_max, text_y_min):
        """Makes row coordinates continuous. For the row to "touch"
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
            [
                max(t.y1 for t in r),
                min(t.y0 for t in r)
            ]
            for r in rows_grouped
        ]
        for i in range(0, len(row_boundaries)-1):
            top_row = row_boundaries[i]
            bottom_row = row_boundaries[i+1]
            top_row[1] = bottom_row[0] = (top_row[1] + bottom_row[0]) / 2
        row_boundaries[0][0] = text_y_max
        row_boundaries[-1][1] = text_y_min
        return row_boundaries

    # FRHTODO: Check if needed, refactor with Stream
    @staticmethod
    def _add_columns(cols, text, row_tol):
        """Add columns to existing list by taking into account
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
            text = Hybrid._group_rows(text, row_tol=row_tol)
            elements = [len(r) for r in text]
            new_cols = [
                (t.x0, t.x1)
                for r in text if len(r) == max(elements)
                for t in r
            ]
            cols.extend(Hybrid._merge_columns(sorted(new_cols)))
        return cols

    # FRHTODO: Check if needed, refactor with Stream
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

    # FRHTODO: Check is needed, refactor with Stream
    def _validate_columns(self):
        if self.table_areas is not None and self.columns is not None:
            if len(self.table_areas) != len(self.columns):
                raise ValueError("Length of table_areas and columns"
                                 " should be equal")

    def _generate_table_bbox(self):
        if self.table_areas is not None:
            table_bbox = {}
            for area_str in self.table_areas:
                table_bbox[bbox_from_str(area_str)] = None
            self.table_bbox = table_bbox
            return

        # Take all the textlines that are not just spaces
        all_textlines = [
            t for t in self.horizontal_text + self.vertical_text
            if len(t.get_text().strip()) > 0
        ]
        textlines = self._apply_regions_filter(all_textlines)

        textlines_processed = {}
        self.table_bbox = {}
        if self.debug_info is not None:
            debug_info_edges_searches = []
            self.debug_info["edges_searches"] = debug_info_edges_searches
            debug_info_bboxes_searches = []
            self.debug_info["bboxes_searches"] = debug_info_bboxes_searches
        else:
            debug_info_edges_searches = None
            debug_info_bboxes_searches = None

        while True:
            self.textedges = TextEdges2()
            self.textedges.generate(textlines)
            self.textedges._remove_unconnected_edges()
            if debug_info_edges_searches is not None:
                # Preserve the current edge calculation for display debugging
                debug_info_edges_searches.append(
                    copy.deepcopy(self.textedges)
                )
            gaps_hv = self.textedges._compute_plausible_gaps()
            if gaps_hv is None:
                return None
            # edge_tol instructions override the calculated vertical gap
            edge_tol_hv = (
                gaps_hv[0],
                gaps_hv[1] if self.edge_tol is None else self.edge_tol
            )
            bbox = self.textedges._build_bbox_candidate(
                edge_tol_hv,
                debug_info=debug_info_bboxes_searches
            )
            if bbox is None:
                break

            # Get all the textlines that are at least 50% in the box
            tls_in_bbox = text_in_bbox(bbox, textlines)

            # and expand the text box to fully contain them
            bbox = bbox_from_text(tls_in_bbox)

            # FRH: do we need to repeat this?
            # tls_in_bbox = text_in_bbox(bbox, textlines)
            cols_anchors = find_columns_coordinates(tls_in_bbox)

            # Apply a heuristic to salvage headers which formatting might be
            # off compared to the rest of the table.
            expanded_bbox = search_header_from_body_bbox(
                bbox,
                textlines,
                cols_anchors,
                gaps_hv[1]
            )

            if self.debug_info is not None:
                if "col_searches" not in self.debug_info:
                    self.debug_info["col_searches"] = []
                self.debug_info["col_searches"].append({
                    "core_bbox": bbox,
                    "cols_anchors": cols_anchors,
                    "expanded_bbox": expanded_bbox
                })

            self.table_bbox[expanded_bbox] = None

            # Remember what textlines we processed, and repeat
            for tl in tls_in_bbox:
                textlines_processed[tl] = None
            textlines = list(filter(
                lambda tl: tl not in textlines_processed,
                textlines
            ))

    # FRHTODO: Check is needed, refactor with Stream
    def _generate_columns_and_rows(self, table_idx, tk):
        # select elements which lie within table_bbox
        self.t_bbox = text_in_bbox_per_axis(
            tk,
            self.horizontal_text,
            self.vertical_text
        )

        text_x_min, text_y_min, text_x_max, text_y_max = \
            self._text_bbox(self.t_bbox)
        rows_grouped = self._group_rows(
            self.t_bbox["horizontal"], row_tol=self.row_tol)
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
                        "No tables found in table area {}"
                        .format(table_idx + 1)
                    )
            cols = [
                (t.x0, t.x1)
                for r in rows_grouped
                if len(r) == ncols
                for t in r
            ]
            cols = self._merge_columns(
                sorted(cols),
                column_tol=self.column_tol
            )
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

    # FRHTODO: Check is needed, refactor with Stream
    def _generate_table(self, table_idx, cols, rows, **kwargs):
        table = self._initialize_new_table(table_idx, cols, rows)
        table = table.set_all_edges()
        table.record_parse_metadata(self)

        # for plotting
        table._bbox = self.table_bbox
        table._segments = None
        table._textedges = self.textedges

        return table

    def extract_tables(self):
        if self._document_has_no_text():
            return []

        # Identify plausible areas within the doc where tables lie,
        # populate table_bbox keys with these areas.
        self._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, bbox in enumerate(
            sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            cols, rows = self._generate_columns_and_rows(table_idx, bbox)
            table = self._generate_table(table_idx, cols, rows)
            table._bbox = bbox
            _tables.append(table)

        return _tables
