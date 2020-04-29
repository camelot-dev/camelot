# -*- coding: utf-8 -*-
"""Implementation of hybrid table parser."""

from __future__ import division

import copy
import math
import numpy as np
import warnings

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
    bbox_from_textlines,
    find_columns_coordinates,
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
    closest = {
        "left": None,
        "right": None,
        "top": None,
        "bottom": None,
    }
    (bbox_left, bbox_bottom, bbox_right, bbox_top) = bbox
    for tl in tls:
        if tl.x1 < bbox_left:
            # Left: check it overlaps horizontally
            if tl.y0 > bbox_top or tl.y1 < bbox_bottom:
                continue
            if closest["left"] is None or closest["left"].x1 < tl.x1:
                closest["left"] = tl
        elif bbox_right < tl.x0:
            # Right: check it overlaps horizontally
            if tl.y0 > bbox_top or tl.y1 < bbox_bottom:
                continue
            if closest["right"] is None or closest["right"].x0 > tl.x0:
                closest["right"] = tl
        else:
            # Either bottom or top: must overlap vertically
            if tl.x0 > bbox_right or tl.x1 < bbox_left:
                continue
            elif tl.y1 < bbox_bottom:
                # Bottom
                if closest["bottom"] is None or closest["bottom"].y1 < tl.y1:
                    closest["bottom"] = tl
            elif bbox_top < tl.y0:
                # Top
                if closest["top"] is None or closest["top"].y0 > tl.y0:
                    closest["top"] = tl
    return closest


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
        for tl in textlines:
            if len(tl.get_text().strip()) > 0:
                self._register_textline(tl)

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

    def _remove_unconnected_edges(self):
        """Weed out elements which are only connected to others vertically
        or horizontally. There needs to be connections across both
        dimensions.
        """
        removed_singletons = True
        while removed_singletons:
            removed_singletons = False
            for textalignments in self._text_alignments.values():
                # For each alignment edge, remove items if they are singletons
                # either horizontally or vertically
                for ta in textalignments:
                    for i in range(len(ta.textlines) - 1, -1, -1):
                        tl = ta.textlines[i]
                        alignments = self._textline_to_alignments[tl]
                        if alignments.max_h_count() <= 1 or \
                           alignments.max_v_count() <= 1:
                            del ta.textlines[i]
                            removed_singletons = True
            self._textline_to_alignments = {}
            self._compute_alignment_counts()

    def most_connected_textline(self):
        """ Retrieve the textline that is most connected across vertical and
        horizontal axis.

        """
        # Find the textline with the highest alignment score
        return max(
            self._textline_to_alignments.keys(),
            key=lambda textline:
            self._textline_to_alignments[textline].alignment_score(),
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
            key=lambda tl: tl.x0,
            reverse=True
        )
        v_textlines = sorted(
            ref_v_textlines,
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

    def _build_bbox_candidate(self, gaps_hv, parse_details=None):
        """ Build a candidate bbox for the body of a table using hybrid algo

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
        last_cols_cand = [most_aligned_tl.x0, most_aligned_tl.x1]
        while last_bbox != bbox:
            if parse_details_search is not None:
                # Store debug info
                parse_details_search["iterations"].append(bbox)

            # Check that the closest tls are within the gaps allowed
            last_bbox = bbox
            cand_bbox = last_bbox.copy()
            closest_tls = find_closest_tls(bbox, tls_search_space)
            for direction, tl in closest_tls.items():
                if tl is None:
                    continue
                expanded_cand_bbox = cand_bbox.copy()

                if direction == "left":
                    if expanded_cand_bbox[0] - tl.x1 > gaps_hv[0]:
                        continue
                    expanded_cand_bbox[0] = tl.x0
                elif direction == "right":
                    if tl.x0 - expanded_cand_bbox[2] > gaps_hv[0]:
                        continue
                    expanded_cand_bbox[2] = tl.x1
                elif direction == "bottom":
                    if expanded_cand_bbox[1] - tl.y1 > gaps_hv[1]:
                        continue
                    expanded_cand_bbox[1] = tl.y0
                elif direction == "top":
                    if tl.y0 - expanded_cand_bbox[3] > gaps_hv[1]:
                        continue
                    expanded_cand_bbox[3] = tl.y1

                # If they are, see what an expanded bbox in that direction
                # would contain
                new_tls = text_in_bbox(expanded_cand_bbox, tls_search_space)
                tls_in_new_box = new_tls + tls_in_bbox

                # And if we're expanding up or down, check that the addition
                # of the new row won't reduce the number of columns.
                # This happens when text covers multiple rows - that's only
                # allowed in the header, treated separately.
                cols_cand = find_columns_coordinates(tls_in_new_box)
                if direction in ["bottom", "top"]:
                    if len(cols_cand) < len(last_cols_cand):
                        continue

                # We have an expansion candidate: register it, update the
                # search space and repeat
                # We use bbox_from_textlines instead of cand_bbox in case some
                # overlapping textlines require a large bbox for strict fit.
                bbox = cand_bbox = list(bbox_from_textlines(tls_in_new_box))
                last_cols_cand = cols_cand
                tls_in_bbox.extend(new_tls)
                for i in range(len(tls_search_space) - 1, -1, -1):
                    tl = tls_search_space[i]
                    if tl in new_tls:
                        del tls_search_space[i]

        if len(tls_in_bbox) > MINIMUM_TEXTLINES_IN_TABLE:
            return bbox
        return None

    def generate(self, textlines):
        """Generate the text edge dictionaries based on the
        input textlines.
        """
        self._register_all_text_lines(textlines)
        self._compute_alignment_counts()


class Hybrid(TextBaseParser):
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
            **kwargs):
        super().__init__(
            "hybrid",
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
        if self.parse_details is not None:
            parse_details_network_searches = []
            self.parse_details["network_searches"] = \
                parse_details_network_searches
            parse_details_bbox_searches = []
            self.parse_details["bbox_searches"] = parse_details_bbox_searches
        else:
            parse_details_network_searches = None
            parse_details_bbox_searches = None

        while True:
            text_network = TextNetworks()
            text_network.generate(textlines)
            text_network._remove_unconnected_edges()
            gaps_hv = text_network._compute_plausible_gaps()
            if gaps_hv is None:
                return None
            # edge_tol instructions override the calculated vertical gap
            edge_tol_hv = (
                gaps_hv[0],
                gaps_hv[1] if self.edge_tol is None else self.edge_tol
            )
            bbox = text_network._build_bbox_candidate(
                edge_tol_hv,
                parse_details=parse_details_bbox_searches
            )
            if bbox is None:
                break

            if parse_details_network_searches is not None:
                # Preserve the current edge calculation for display debugging
                parse_details_network_searches.append(
                    copy.deepcopy(text_network)
                )

            # Get all the textlines that are at least 50% in the box
            tls_in_bbox = text_in_bbox(bbox, textlines)

            # and expand the text box to fully contain them
            bbox = bbox_from_textlines(tls_in_bbox)

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

            if self.parse_details is not None:
                if "col_searches" not in self.parse_details:
                    self.parse_details["col_searches"] = []
                self.parse_details["col_searches"].append({
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

    def _generate_columns_and_rows(self, bbox, table_idx):
        # select elements which lie within table_bbox
        self.t_bbox = text_in_bbox_per_axis(
            bbox,
            self.horizontal_text,
            self.vertical_text
        )

        all_tls = list(
            sorted(
                filter(
                    lambda tl: len(tl.get_text().strip()) > 0,
                    self.t_bbox["horizontal"] + self.t_bbox["vertical"]
                ),
                key=lambda tl: (-tl.y0, tl.x0)
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

        return cols, rows, None, None
