# -*- coding: utf-8 -*-
"""Implementation of hybrid table parser."""

from __future__ import division

import numpy as np
import copy

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
    distance_tl_to_bbox,
    find_columns_coordinates
)

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


class AlignmentCounter(object):
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
        self._textlines_alignments = {}

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
                    alignments = self._textlines_alignments.get(
                        textline, None)
                    if alignments is None:
                        alignments = AlignmentCounter()
                        self._textlines_alignments[textline] = alignments
                    alignments[align_id] = textedge.textlines

    def _calculate_gaps_thresholds(self, percentile=75):
        """Identify reasonable gaps between lines and columns based
        on gaps observed across alignments.
        This can be used to reject cells as too far away from
        the core table.
        """
        h_gaps, v_gaps = [], []
        for align_id in self._text_alignments:
            edge_array = self._text_alignments[align_id]
            gaps = []
            vertical = align_id in HORIZONTAL_ALIGNMENTS
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
                    f"for {align_id}: "
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
            for alignment_id, textalignments in self._text_alignments.items():
                # For each alignment edge, remove items if they are singletons
                # either horizontally or vertically
                for ta in textalignments:
                    for i in range(len(ta.textlines) - 1, -1, -1):
                        tl = ta.textlines[i]
                        alignments = self._textlines_alignments[tl]
                        if alignments.max_h_count() <= 1 or \
                           alignments.max_v_count() <= 1:
                            del ta.textlines[i]
                            removed_singletons = True
            self._textlines_alignments = {}
            self._compute_alignment_counts()

    def most_connected_textline(self):
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
        # Determine the textline that has the most combined
        # alignments across horizontal and vertical axis.
        # It will serve as a reference axis along which to collect the average
        # spacing between rows/cols.
        most_aligned_tl = self.most_connected_textline()
        if most_aligned_tl is None:
            return None

        # Retrieve the list of textlines it's aligned with, across both
        # axis
        best_alignment = self._textlines_alignments[most_aligned_tl]
        ref_h_alignment_id, ref_h_textlines = best_alignment.max_h()
        ref_v_alignment_id, ref_v_textlines = best_alignment.max_v()
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
        """ Seed the process with the textline with the highest alignment
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
            if parse_details_search is not None:
                # Store debug info
                parse_details_search["iterations"].append(bbox)

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
        **kwargs
    ):
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
