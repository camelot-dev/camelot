# -*- coding: utf-8 -*-
"""Implementation of network table parser."""

from __future__ import division

import copy
import math
import numpy as np

from .base import TextBaseParser
from ..core import (
    TextAlignments,
    ALL_ALIGNMENTS,
    HORIZONTAL_ALIGNMENTS,
    VERTICAL_ALIGNMENTS
)
from ..utils import (
    bbox_from_str,
    text_in_bbox,
    textlines_overlapping_bbox,
    bbox_from_textlines,
    find_columns_boundaries,
    boundaries_to_split_lines,
    text_in_bbox_per_axis,
)

# maximum number of columns over which a header can spread
MAX_COL_SPREAD_IN_HEADER = 3

# Minimum number of textlines in a table
MINIMUM_TEXTLINES_IN_TABLE = 6


def column_spread(left, right, col_anchors):
    """Get the number of columns crossed by a segment [left, right]."""
    index_left = 0
    while index_left < len(col_anchors) \
            and col_anchors[index_left] < left:
        index_left += 1
    index_right = index_left
    while index_right < len(col_anchors) \
            and col_anchors[index_right] < right:
        index_right += 1

    return index_right - index_left


def find_closest_tls(bbox, tls):
    """ Search for tls that are the closest but outside in all 4 directions
    """
    left, right, top, bottom = None, None, None, None
    (bbox_left, bbox_bottom, bbox_right, bbox_top) = bbox
    for textline in tls:
        if textline.x1 < bbox_left:
            # Left: check it overlaps horizontally
            if textline.y0 > bbox_top or textline.y1 < bbox_bottom:
                continue
            if left is None or left.x1 < textline.x1:
                left = textline
        elif bbox_right < textline.x0:
            # Right: check it overlaps horizontally
            if textline.y0 > bbox_top or textline.y1 < bbox_bottom:
                continue
            if right is None or right.x0 > textline.x0:
                right = textline
        else:
            # Either bottom or top: must overlap vertically
            if textline.x0 > bbox_right or textline.x1 < bbox_left:
                continue
            if textline.y1 < bbox_bottom:
                # Bottom
                if bottom is None or bottom.y1 < textline.y1:
                    bottom = textline
            elif bbox_top < textline.y0:
                # Top
                if top is None or top.y0 > textline.y0:
                    top = textline
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


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

    keep_searching = True
    while keep_searching:
        keep_searching = False
        # a/ first look for the closest text element above the bbox.
        # It will be the anchor for a possible new row.
        closest_above = None
        all_above = []
        for textline in textlines:
            # higher than the table, >50% within its bounds
            textline_center = 0.5 * (textline.x0 + textline.x1)
            if textline.y0 > top and left < textline_center < right:
                all_above.append(textline)
                if closest_above is None or closest_above.y0 > textline.y0:
                    closest_above = textline

        if closest_above and closest_above.y0 < top + max_v_gap:
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
                    textline = all_above[i]
                    if textline.y0 < top:
                        # The bottom of this element is within our row
                        # so we add it.
                        tls_in_new_row.append(textline)
                        all_above.pop(i)
                        if textline.y1 > top:
                            # If the top of this element raises our row's
                            # band, we'll need to keep on searching for
                            # overlapping items
                            top = textline.y1
                            pushed_up = True

            # Get the x-ranges for all the textlines, and merge the
            # x-ranges that overlap
            zones = zones + list(
                map(
                    lambda textline: [textline.x0, textline.x1],
                    tls_in_new_row
                )
            )
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

            # Accept textlines that cross columns boundaries, as long as they
            # cross less than MAX_COL_SPREAD_IN_HEADER, and half the number of
            # columns.
            # This is to avoid picking unrelated paragraphs.
            if max_spread <= min(
                    MAX_COL_SPREAD_IN_HEADER,
                    math.ceil(len(col_anchors) / 2)):
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


class AlignmentCounter():
    """
    For a given textline, represent all other textlines aligned with it.

    A textline can be vertically aligned with others if their bbox match on
    left, right, or middle coord, and horizontally aligned if they match top,
    bottom, or center coord.

    """

    def __init__(self):
        self.alignment_to_occurrences = {}
        for alignment in ALL_ALIGNMENTS:
            self.alignment_to_occurrences[alignment] = []

    def __getitem__(self, key):
        return self.alignment_to_occurrences[key]

    def __setitem__(self, key, value):
        self.alignment_to_occurrences[key] = value
        return value

    def max_alignments(self, alignment_ids=None):
        """Get the alignment dimension with the max number of textlines.

        """
        alignment_ids = alignment_ids or self.alignment_to_occurrences.keys()
        alignment_items = map(
            lambda alignment_id: (
                alignment_id,
                self.alignment_to_occurrences[alignment_id]
            ),
            alignment_ids
        )
        return max(alignment_items, key=lambda item: len(item[1]))

    def max_v(self):
        """Tuple (alignment_id, textlines) of largest vertical row.
        """
        # Note that the horizontal alignments (left, center, right) are aligned
        # vertically in a column, so max_v is calculated by looking at
        # horizontal alignments.
        return self.max_alignments(HORIZONTAL_ALIGNMENTS)

    def max_h(self):
        """Tuple (alignment_id, textlines) of largest horizontal col.
        """
        return self.max_alignments(VERTICAL_ALIGNMENTS)

    def max_v_count(self):
        """Returns the maximum number of alignments along
        one of the vertical axis (left/right/middle).
        """
        return len(self.max_v()[1])

    def max_h_count(self):
        """Returns the maximum number of alignments along
        one of the horizontal axis (bottom/top/center).
        """
        return len(self.max_h()[1])

    def alignment_score(self):
        """We define the alignment score of a textline as the product of the
        number of aligned elements - 1. The -1 is to avoid favoring
         singletons on a long line.
        """
        return (self.max_v_count()-1) * (self.max_h_count()-1)


class TextNetworks(TextAlignments):
    """Text elements connected by vertical AND horizontal alignments.

    The alignment dict has six keys based on the hor/vert alignments,
    and each key's value is a list of camelot.core.TextAlignment objects.
    """

    def __init__(self):
        super().__init__(ALL_ALIGNMENTS)
        # For each textline, dictionary "alignment type" to
        # "number of textlines aligned"
        self._textline_to_alignments = {}

    def _update_alignment(self, alignment, coord, textline):
        alignment.register_aligned_textline(textline, coord)

    def _register_all_text_lines(self, textlines):
        """Add all textlines to our network repository to
        identify alignments.
        """
        # Identify all the alignments
        for textline in textlines:
            if len(textline.get_text().strip()) > 0:
                self._register_textline(textline)

    def _compute_alignment_counts(self):
        """Build a dictionary textline -> alignment object.
        """
        for align_id, textedges in self._text_alignments.items():
            for textedge in textedges:
                for textline in textedge.textlines:
                    alignments = self._textline_to_alignments.get(
                        textline, None)
                    if alignments is None:
                        alignments = AlignmentCounter()
                        self._textline_to_alignments[textline] = alignments
                    alignments[align_id] = textedge.textlines

    def remove_unconnected_edges(self):
        """Weed out elements which are only connected to others vertically
        or horizontally. There needs to be connections across both
        dimensions.
        """
        removed_singletons = True
        while removed_singletons:
            removed_singletons = False
            for text_alignments in self._text_alignments.values():
                # For each alignment edge, remove items if they are singletons
                # either horizontally or vertically
                for text_alignment in text_alignments:
                    for i in range(len(text_alignment.textlines) - 1, -1, -1):
                        textline = text_alignment.textlines[i]
                        alignments = self._textline_to_alignments[textline]
                        if alignments.max_h_count() <= 1 or \
                           alignments.max_v_count() <= 1:
                            del text_alignment.textlines[i]
                            removed_singletons = True
            self._textline_to_alignments = {}
            self._compute_alignment_counts()

    def most_connected_textline(self):
        """ Retrieve the textline that is most connected across vertical and
        horizontal axis.

        """
        # Find the textline with the highest alignment score, with a tie break
        # to prefer textlines further down in the table.  Starting the search
        # from the table's bottom allows the algo to collect data on more cells
        # before going to the header, typically harder to parse.
        return max(
            self._textline_to_alignments.keys(),
            key=lambda textline:
            (
                self._textline_to_alignments[textline].alignment_score(),
                -textline.y0, -textline.x0
            ),
            default=None
        )

    def compute_plausible_gaps(self):
        """ Evaluate plausible gaps between cells horizontally and vertically
        based on the textlines aligned with the most connected textline.

        Returns
        -------
        gaps_hv : tuple
            (horizontal_gap, horizontal_gap) in pdf coordinate space.

        """
        # Determine the textline that has the most combined
        # alignments across horizontal and vertical axis.
        # It will serve as a reference axis along which to collect the average
        # spacing between rows/cols.
        most_aligned_tl = self.most_connected_textline()
        if most_aligned_tl is None:
            return None

        # Retrieve the list of textlines it's aligned with, across both
        # axis
        best_alignment = self._textline_to_alignments[most_aligned_tl]
        __, ref_h_textlines = best_alignment.max_h()
        __, ref_v_textlines = best_alignment.max_v()
        if len(ref_v_textlines) <= 1 or len(ref_h_textlines) <= 1:
            return None

        h_textlines = sorted(
            ref_h_textlines,
            key=lambda textline: textline.x0,
            reverse=True
        )
        v_textlines = sorted(
            ref_v_textlines,
            key=lambda textline: textline.y0,
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

    def search_table_body(self, gaps_hv, parse_details=None):
        """ Build a candidate bbox for the body of a table using network algo

        Seed the process with the textline with the highest alignment
        score, then expand the bbox with textlines within threshold.

        Parameters
        ----------
        gaps_hv : tuple
            The maximum distance allowed to consider surrounding lines/columns
            as part of the same table.
        parse_details : array (optional)
            Optional parameter array, in which to store extra information
            to help later visualization of the table creation.
        """
        # First, determine the textline that has the most combined
        # alignments across horizontal and vertical axis.
        # It will serve both as a starting point for the table boundary
        # search, and as a way to estimate the average spacing between
        # rows/cols.
        most_aligned_tl = self.most_connected_textline()

        # Calculate the 75th percentile of the horizontal/vertical
        # gaps between textlines.  Use this as a reference for a threshold
        # to not exceed while looking for table boundaries.
        max_h_gap, max_v_gap = gaps_hv[0], gaps_hv[1]

        if parse_details is not None:
            # Store debug info
            parse_details_search = {
                "max_h_gap": max_h_gap,
                "max_v_gap": max_v_gap,
                "iterations": []
            }
            parse_details.append(parse_details_search)
        else:
            parse_details_search = None

        bbox = [most_aligned_tl.x0, most_aligned_tl.y0,
                most_aligned_tl.x1, most_aligned_tl.y1]

        # For the body of the table, we only consider cells that have
        # alignments on both axis.
        tls_search_space = list(self._textline_to_alignments.keys())
        # tls_search_space = []
        tls_search_space.remove(most_aligned_tl)
        tls_in_bbox = [most_aligned_tl]
        last_bbox = None
        last_cols_bounds = [(most_aligned_tl.x0, most_aligned_tl.x1)]
        while last_bbox != bbox:
            if parse_details_search is not None:
                # Store debug info
                parse_details_search["iterations"].append(bbox)

            # Check that the closest tls are within the gaps allowed
            last_bbox = bbox
            cand_bbox = last_bbox.copy()
            closest_tls = find_closest_tls(bbox, tls_search_space)
            for direction, textline in closest_tls.items():
                if textline is None:
                    continue
                expanded_cand_bbox = cand_bbox.copy()

                if direction == "left":
                    if expanded_cand_bbox[0] - textline.x1 > gaps_hv[0]:
                        continue
                    expanded_cand_bbox[0] = textline.x0
                elif direction == "right":
                    if textline.x0 - expanded_cand_bbox[2] > gaps_hv[0]:
                        continue
                    expanded_cand_bbox[2] = textline.x1
                elif direction == "bottom":
                    if expanded_cand_bbox[1] - textline.y1 > gaps_hv[1]:
                        continue
                    expanded_cand_bbox[1] = textline.y0
                elif direction == "top":
                    if textline.y0 - expanded_cand_bbox[3] > gaps_hv[1]:
                        continue
                    expanded_cand_bbox[3] = textline.y1

                # If they are, see what an expanded bbox in that direction
                # would contain
                new_tls = text_in_bbox(expanded_cand_bbox, tls_search_space)
                tls_in_new_box = new_tls + tls_in_bbox

                # And if we're expanding up or down, check that the addition
                # of the new row won't reduce the number of columns.
                # This happens when text covers multiple rows - that's only
                # allowed in the header, treated separately.
                cols_bounds = find_columns_boundaries(tls_in_new_box)
                if direction in ["bottom", "top"] and \
                        len(cols_bounds) < len(last_cols_bounds):
                    continue

                # We have an expansion candidate: register it, update the
                # search space and repeat
                # We use bbox_from_textlines instead of cand_bbox in case some
                # overlapping textlines require a large bbox for strict fit.
                bbox = cand_bbox = list(bbox_from_textlines(tls_in_new_box))
                last_cols_bounds = cols_bounds
                tls_in_bbox.extend(new_tls)
                for i in range(len(tls_search_space) - 1, -1, -1):
                    textline = tls_search_space[i]
                    if textline in new_tls:
                        del tls_search_space[i]

        if len(tls_in_bbox) >= MINIMUM_TEXTLINES_IN_TABLE:
            return bbox
        return None

    def generate(self, textlines):
        """Generate the text edge dictionaries based on the
        input textlines.
        """
        self._register_all_text_lines(textlines)
        self._compute_alignment_counts()


class Network(TextBaseParser):
    """Network method of parsing looks for spaces between text
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
            **kwargs):
        super().__init__(
            "network",
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

    def _generate_table_bbox(self):
        user_provided_bboxes = None
        if self.table_areas is not None:
            # User gave us table areas already.  We will use their coordinates
            # to find column anchors.
            user_provided_bboxes = []
            for area_str in self.table_areas:
                user_provided_bboxes.append(bbox_from_str(area_str))

        # Take all the textlines that are not just spaces
        all_textlines = [
            t for t in self.horizontal_text + self.vertical_text
            if len(t.get_text().strip()) > 0
        ]
        textlines = self._apply_regions_filter(all_textlines)

        textlines_processed = {}
        self.table_bbox_parses = {}
        if self.parse_details is not None:
            parse_details_network_searches = []
            self.parse_details["network_searches"] = \
                parse_details_network_searches
            parse_details_bbox_searches = []
            self.parse_details["bbox_searches"] = parse_details_bbox_searches
            self.parse_details["col_searches"] = []
        else:
            parse_details_network_searches = None
            parse_details_bbox_searches = None

        while True:
            # Find a bbox: either pulling from the user's or from the network
            # algorithm.

            # First look for the body of the table
            bbox_body = None
            if user_provided_bboxes is not None:
                if len(user_provided_bboxes) > 0:
                    bbox_body = user_provided_bboxes.pop()
            else:
                text_network = TextNetworks()
                text_network.generate(textlines)
                text_network.remove_unconnected_edges()
                gaps_hv = text_network.compute_plausible_gaps()
                if gaps_hv is None:
                    return None
                # edge_tol instructions override the calculated vertical gap
                edge_tol_hv = (
                    gaps_hv[0],
                    gaps_hv[1] if self.edge_tol is None else self.edge_tol
                )
                bbox_body = text_network.search_table_body(
                    edge_tol_hv,
                    parse_details=parse_details_bbox_searches
                )

                if parse_details_network_searches is not None:
                    # Preserve the current edge calculation for debugging
                    parse_details_network_searches.append(
                        copy.deepcopy(text_network)
                    )

            if bbox_body is None:
                break

            # Get all the textlines that overlap with the box, compute
            # columns
            tls_in_bbox = textlines_overlapping_bbox(bbox_body, textlines)
            cols_boundaries = find_columns_boundaries(tls_in_bbox)
            cols_anchors = boundaries_to_split_lines(cols_boundaries)

            # Unless the user gave us strict bbox_body, try to find a header
            # above the body to build the full bbox.
            if user_provided_bboxes is not None:
                bbox_full = bbox_body
            else:
                # Expand the text box to fully contain the tls we found
                bbox_body = bbox_from_textlines(tls_in_bbox)

                # Apply a heuristic to salvage headers which formatting might
                # be off compared to the rest of the table.
                bbox_full = search_header_from_body_bbox(
                    bbox_body,
                    textlines,
                    cols_anchors,
                    gaps_hv[1]
                )

            table_parse = {
                "bbox_body": bbox_body,
                "cols_boundaries": cols_boundaries,
                "cols_anchors": cols_anchors,
                "bbox_full": bbox_full
            }
            self.table_bbox_parses[bbox_full] = table_parse

            if self.parse_details is not None:
                self.parse_details["col_searches"].append(table_parse)

            # Remember what textlines we processed, and repeat
            for textline in tls_in_bbox:
                textlines_processed[textline] = None
            textlines = list(filter(
                lambda textline: textline not in textlines_processed,
                textlines
            ))

    def _generate_columns_and_rows(self, bbox, user_cols):
        # select elements which lie within table_bbox
        self.t_bbox = text_in_bbox_per_axis(
            bbox,
            self.horizontal_text,
            self.vertical_text
        )

        all_tls = list(
            sorted(
                filter(
                    lambda textline: len(textline.get_text().strip()) > 0,
                    self.t_bbox["horizontal"] + self.t_bbox["vertical"]
                ),
                key=lambda textline: (-textline.y0, textline.x0)
            )
        )
        text_x_min, text_y_min, text_x_max, text_y_max = bbox_from_textlines(
            all_tls
        )
        # FRHTODO:
        # This algorithm takes the horizontal textlines in the bbox, and groups
        # them into rows based on their bottom y0.
        # That's wrong: it misses the vertical items, and misses out on all
        # the alignment identification work we've done earlier.
        rows_grouped = self._group_rows(all_tls, row_tol=self.row_tol)
        rows = self._join_rows(rows_grouped, text_y_max, text_y_min)

        if user_cols is not None:
            cols = [text_x_min] + user_cols + [text_x_max]
            cols = [
                (cols[i], cols[i + 1])
                for i in range(0, len(cols) - 1)
            ]
        else:
            parse_details = self.table_bbox_parses[bbox]
            col_anchors = parse_details["cols_anchors"]
            cols = list(map(
                lambda idx: [col_anchors[idx], col_anchors[idx + 1]],
                range(0, len(col_anchors) - 1)
            ))

        return cols, rows, None, None
