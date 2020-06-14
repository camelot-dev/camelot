# -*- coding: utf-8 -*-

import os
import atexit
import re
import random
import shutil
import string
import tempfile
import warnings
from itertools import groupby
from operator import itemgetter
from urllib.request import Request

import numpy as np
import pandas as pd
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTTextLineHorizontal,
    LTTextLineVertical,
    LTImage,
)

from .ext.ghostscript import Ghostscript

from urllib.request import urlopen
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_relative, uses_netloc, uses_params


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


# https://github.com/pandas-dev/pandas/blob/master/pandas/io/common.py
def is_url(url):
    """Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If url has a valid protocol return True otherwise False.

    """
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False


def random_string(length):
    ret = ""
    while length:
        ret += random.choice(
            string.digits + string.ascii_lowercase + string.ascii_uppercase
        )
        length -= 1
    return ret


def download_url(url):
    """Download file from specified URL.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    filepath : str or unicode
        Temporary filepath.

    """
    filename = f"{random_string(6)}.pdf"
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        headers = {"User-Agent": "Mozilla/5.0"}
        request = Request(url, None, headers)
        obj = urlopen(request)
        content_type = obj.info().get_content_type()
        if content_type != "application/pdf":
            raise NotImplementedError("File format not supported")
        f.write(obj.read())
    filepath = os.path.join(os.path.dirname(f.name), filename)
    shutil.move(f.name, filepath)
    return filepath


common_kwargs = [
    "flag_size",
    "margins",
    "split_text",
    "strip_text",
    "table_areas",
    "table_regions"
]
text_kwargs = common_kwargs + [
    "columns",
    "edge_tol",
    "row_tol",
    "column_tol"
]
lattice_kwargs = common_kwargs + [
    "process_background",
    "line_scale",
    "copy_text",
    "shift_text",
    "line_tol",
    "joint_tol",
    "threshold_blocksize",
    "threshold_constant",
    "iterations",
    "resolution",
]
flavor_to_kwargs = {
    "stream": text_kwargs,
    "network": text_kwargs,
    "lattice": lattice_kwargs,
    "hybrid": text_kwargs + lattice_kwargs,
}


def validate_input(kwargs, flavor="lattice"):
    parser_kwargs = flavor_to_kwargs[flavor]
    # s.difference(t): new set with elements in s but not in t
    isec = set(kwargs.keys()).difference(set(parser_kwargs))
    if isec:
        raise ValueError(
            f"{','.join(sorted(isec))} cannot be used with flavor='{flavor}'"
        )


def remove_extra(kwargs, flavor="lattice"):
    parser_kwargs = flavor_to_kwargs[flavor]
    # Avoid "dictionary changed size during iteration"
    kwargs_keys = list(kwargs.keys())
    for key in kwargs_keys:
        if key not in parser_kwargs:
            kwargs.pop(key)
    return kwargs


# https://stackoverflow.com/a/22726782
# and https://stackoverflow.com/questions/10965479
class TemporaryDirectory():
    def __init__(self):
        self.dir_path = None

    def __enter__(self):
        self.dir_path = tempfile.mkdtemp()
        # Only delete the temporary directory upon
        # program exit.
        atexit.register(shutil.rmtree, self.dir_path)
        return self.dir_path

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def build_file_path_in_temp_dir(filename, extension=None):
    """Generates a new path within a temporary directory

    Parameters
    ----------
    filename : str
    extension : str

    Returns
    -------
    file_path_in_temporary_dir : str

    """
    with TemporaryDirectory() as temp_dir:
        if extension:
            filename = filename + extension
        path = os.path.join(
            temp_dir,
            filename
        )
    return path


def translate(x1, x2):
    """Translates x2 by x1.

    Parameters
    ----------
    x1 : float
    x2 : float

    Returns
    -------
    x2 : float

    """
    x2 += x1
    return x2


def scale(x, s):
    """Scales x by scaling factor s.

    Parameters
    ----------
    x : float
    s : float

    Returns
    -------
    x : float

    """
    x *= s
    return x


def scale_pdf(k, factors):
    """Translates and scales pdf coordinate space to image
    coordinate space.

    Parameters
    ----------
    k : tuple
        Tuple (x1, y1, x2, y2) representing table bounding box where
        (x1, y1) -> lt and (x2, y2) -> rb in PDFMiner coordinate
        space.
    factors : tuple
        Tuple (scaling_factor_x, scaling_factor_y, pdf_y) where the
        first two elements are scaling factors and pdf_y is height of
        pdf.

    Returns
    -------
    knew : tuple
        Tuple (x1, y1, x2, y2) representing table bounding box where
        (x1, y1) -> lt and (x2, y2) -> rb in OpenCV coordinate
        space.

    """
    x1, y1, x2, y2 = k
    scaling_factor_x, scaling_factor_y, pdf_y = factors
    x1 = scale(x1, scaling_factor_x)
    y1 = scale(abs(translate(-pdf_y, y1)), scaling_factor_y)
    x2 = scale(x2, scaling_factor_x)
    y2 = scale(abs(translate(-pdf_y, y2)), scaling_factor_y)
    knew = (int(x1), int(y1), int(x2), int(y2))
    return knew


def scale_image(tables, v_segments, h_segments, factors):
    """Translates and scales image coordinate space to pdf
    coordinate space.

    Parameters
    ----------
    tables : dict
        Dict with table boundaries as keys and list of intersections
        in that boundary as value.
    v_segments : list
        List of vertical line segments.
    h_segments : list
        List of horizontal line segments.
    factors : tuple
        Tuple (scaling_factor_x, scaling_factor_y, img_y) where the
        first two elements are scaling factors and img_y is height of
        image.

    Returns
    -------
    tables_new : dict
    v_segments_new : dict
    h_segments_new : dict

    """
    scaling_factor_x, scaling_factor_y, img_y = factors
    tables_new = {}
    for k in tables.keys():
        x1, y1, x2, y2 = k
        x1 = scale(x1, scaling_factor_x)
        y1 = scale(abs(translate(-img_y, y1)), scaling_factor_y)
        x2 = scale(x2, scaling_factor_x)
        y2 = scale(abs(translate(-img_y, y2)), scaling_factor_y)
        j_x, j_y = zip(*tables[k])
        j_x = [scale(j, scaling_factor_x) for j in j_x]
        j_y = [scale(abs(translate(-img_y, j)), scaling_factor_y) for j in j_y]
        tables_new[(x1, y1, x2, y2)] = {
            "joints": list(zip(j_x, j_y))
        }

    v_segments_new = []
    for v in v_segments:
        x1, x2 = scale(v[0], scaling_factor_x), scale(v[2], scaling_factor_x)
        y1, y2 = (
            scale(abs(translate(-img_y, v[1])), scaling_factor_y),
            scale(abs(translate(-img_y, v[3])), scaling_factor_y),
        )
        v_segments_new.append((x1, y1, x2, y2))

    h_segments_new = []
    for h in h_segments:
        x1, x2 = scale(h[0], scaling_factor_x), scale(h[2], scaling_factor_x)
        y1, y2 = (
            scale(abs(translate(-img_y, h[1])), scaling_factor_y),
            scale(abs(translate(-img_y, h[3])), scaling_factor_y),
        )
        h_segments_new.append((x1, y1, x2, y2))

    return tables_new, v_segments_new, h_segments_new


def get_rotation(chars, horizontal_text, vertical_text):
    """Detects if text in table is rotated or not using the current
    transformation matrix (CTM) and returns its orientation.

    Parameters
    ----------
    horizontal_text : list
        List of PDFMiner LTTextLineHorizontal objects.
    vertical_text : list
        List of PDFMiner LTTextLineVertical objects.
    ltchar : list
        List of PDFMiner LTChar objects.

    Returns
    -------
    rotation : string
        '' if text in table is upright, 'anticlockwise' if
        rotated 90 degree anticlockwise and 'clockwise' if
        rotated 90 degree clockwise.

    """
    rotation = ""
    hlen = len([t for t in horizontal_text if t.get_text().strip()])
    vlen = len([t for t in vertical_text if t.get_text().strip()])
    if hlen < vlen:
        clockwise = sum(t.matrix[1] < 0 < t.matrix[2] for t in chars)
        anticlockwise = sum(t.matrix[1] > 0 > t.matrix[2] for t in chars)
        rotation = "anticlockwise" if clockwise < anticlockwise \
            else "clockwise"
    return rotation


def segments_in_bbox(bbox, v_segments, h_segments):
    """Returns all line segments present inside a bounding box.

    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
        space.
    v_segments : list
        List of vertical line segments.
    h_segments : list
        List of vertical horizontal segments.

    Returns
    -------
    v_s : list
        List of vertical line segments that lie inside table.
    h_s : list
        List of horizontal line segments that lie inside table.

    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    v_s = [
        v
        for v in v_segments
        if v[1] > lb[1] - 2 and
        v[3] < rt[1] + 2 and
        lb[0] - 2 <= v[0] <= rt[0] + 2
    ]
    h_s = [
        h
        for h in h_segments
        if h[0] > lb[0] - 2 and
        h[2] < rt[0] + 2 and
        lb[1] - 2 <= h[1] <= rt[1] + 2
    ]
    return v_s, h_s


def get_textline_coords(textline):
    """Calculate the coordinates of each alignment for a given textline.
    """
    return {
        "left": textline.x0,
        "right": textline.x1,
        "middle": (textline.x0 + textline.x1) / 2.0,
        "bottom": textline.y0,
        "top": textline.y1,
        "center": (textline.y0 + textline.y1) / 2.0,
    }


def bbox_from_str(bbox_str):
    """Deserialize bbox from string ("x1,y1,x2,y2") to tuple (x1, y1, x2, y2).

    Parameters
    ----------
    bbox_str : str
        Serialized bbox with comma separated coordinates, "x1,y1,x2,y2".

    Returns
    -------
    bbox : tuple
        Tuple (x1, y1, x2, y2).

    """
    x1, y1, x2, y2 = bbox_str.split(",")
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    return (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2)
    )


def bboxes_overlap(bbox1, bbox2):
    (left1, bottom1, right1, top1) = bbox1
    (left2, bottom2, right2, top2) = bbox2
    return (
            (left1 < left2 < right1) or (left1 < right2 < right1)
        ) and (
            (bottom1 < bottom2 < top1) or (bottom1 < top2 < top1)
        )


def textlines_overlapping_bbox(bbox, textlines):
    """Returns all text objects which overlap or are within a bounding box.

    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.
    textlines : List of PDFMiner text objects.

    Returns
    -------
    t_bbox : list
        List of PDFMiner text objects.

    """
    t_bbox = [
        t
        for t in textlines
        if bboxes_overlap(bbox, (t.x0, t.y0, t.x1, t.y1))
    ]
    return t_bbox


def text_in_bbox(bbox, text):
    """Returns all text objects which lie at least 50% inside a bounding box
    across both dimensions.

    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.
    text : List of PDFMiner text objects.

    Returns
    -------
    t_bbox : list
        List of PDFMiner text objects that lie inside table.

    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    t_bbox = [
        t
        for t in text
        if lb[0] - 2 <= (t.x0 + t.x1) / 2.0 <= rt[0] + 2
        and lb[1] - 2 <= (t.y0 + t.y1) / 2.0 <= rt[1] + 2
    ]
    return t_bbox


def text_in_bbox_per_axis(bbox, horizontal_text, vertical_text):
    """Returns all text objects present inside a bounding box, split between
    horizontal and vertical text.

    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.
    horizontal_text : List of PDFMiner text objects.
    vertical_text : List of PDFMiner text objects.

    Returns
    -------
    t_bbox : dict
        Dict of lists of PDFMiner text objects that lie inside table, with one
        key each for "horizontal" and "vertical"

    """
    t_bbox = {}
    t_bbox["horizontal"] = text_in_bbox(bbox, horizontal_text)
    t_bbox["vertical"] = text_in_bbox(bbox, vertical_text)

    t_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
    t_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))
    return t_bbox


def expand_bbox_with_textline(bbox, textline):
    """Expand (if needed) a bbox so that it fits the parameter textline.
    """
    return (
        min(bbox[0], textline.x0),
        min(bbox[1], textline.y0),
        max(bbox[2], textline.x1),
        max(bbox[3], textline.y1)
    )


def bbox_from_textlines(textlines):
    """Returns the smallest bbox containing all the text objects passed as
    a parameters.

    Parameters
    ----------
    textlines : List of PDFMiner text objects.

    Returns
    -------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.

    """
    if len(textlines) == 0:
        return None
    bbox = (
        textlines[0].x0,
        textlines[0].y0,
        textlines[0].x1,
        textlines[0].y1
    )

    for tl in textlines[1:]:
        bbox = expand_bbox_with_textline(bbox, tl)
    return bbox


def find_columns_boundaries(tls, min_gap=1.0):
    """Make a list of disjunct cols boundaries for a list of text objects

    Parameters
    ----------
    tls : list of PDFMiner text object.

    min_gap : minimum distance between columns. Any elements closer than
        this threshold are merged together.  This is to prevent spaces between
        words to be misinterpreted as boundaries.

    Returns
    -------
    boundaries : list
        List x-coordinates for cols.
         [(1st col left, 1st col right), (2nd col left, 2nd col right), ...]


    """
    cols_bounds = []
    tls.sort(key=lambda tl: tl.x0)
    for tl in tls:
        if (not cols_bounds) or cols_bounds[-1][1] + min_gap < tl.x0:
            cols_bounds.append([tl.x0, tl.x1])
        else:
            cols_bounds[-1][1] = max(cols_bounds[-1][1], tl.x1)
    return cols_bounds


def find_rows_boundaries(tls, min_gap=1.0):
    """Make a list of disjunct rows boundaries for a list of text objects

    Parameters
    ----------
    tls : list of PDFMiner text object.

    min_gap : minimum distance between rows. Any elements closer than
        this threshold are merged together.

    Returns
    -------
    boundaries : list
        List y-coordinates for rows.
         [(1st row bottom, 1st row top), (2nd row bottom, 2nd row top), ...]

    """
    rows_bounds = []
    tls.sort(key=lambda tl: tl.y0)
    for tl in tls:
        if (not rows_bounds) or rows_bounds[-1][1] + min_gap < tl.y0:
            rows_bounds.append([tl.y0, tl.y1])
        else:
            rows_bounds[-1][1] = max(rows_bounds[-1][1], tl.y1)
    return rows_bounds


def boundaries_to_split_lines(boundaries):
    """Find split lines given a list of boundaries between rows or cols.

    Boundaries:     [ a ]         [b]     [   c   ]  [d]
    Splits:         |        |         |            |  |

    Parameters
    ----------
    boundaries : list
        List of tuples of x- (for columns) or y- (for rows) coord boundaries.
        These are the (left, right most) or (bottom, top most) coordinates.

    Returns
    -------
    anchors : list
        List of coordinates representing the split points, each half way
        between boundaries

    """
    # From the row boundaries, identify splits by getting the mid points
    # between the boundaries.
    anchors = list(map(
        lambda idx: (boundaries[idx-1][1] + boundaries[idx][0]) / 2.0,
        range(1, len(boundaries))
    ))
    anchors.insert(0, boundaries[0][0])
    anchors.append(boundaries[-1][1])
    return anchors


def get_index_closest_point(point, sorted_list, fn=lambda x: x):
    """Return the index of the closest point in the sorted list.

    Parameters
    ----------
    point : the reference sortable element to search.
    sorted_list : list
    fn: optional accessor function

    Returns
    -------
    index : int

    """
    n = len(sorted_list)
    if n == 0:
        return None
    if n == 1:
        return 0

    left = 0
    right = n - 1
    mid = 0

    if point >= fn(sorted_list[n - 1]):
        return n - 1
    if point <= fn(sorted_list[0]):
        return 0

    while left < right:
        mid = (left + right) // 2  # find the mid
        mid_val = fn(sorted_list[mid])
        if point < mid_val:
            right = mid
        elif point > mid_val:
            left = mid + 1
        else:
            return mid

    if mid_val > point:
        if mid > 0 and (
                point - fn(sorted_list[mid-1]) <
                mid_val - point):
            return mid-1
    elif mid_val < point:
        if mid < n - 1 and (
                fn(sorted_list[mid+1]) - point <
                point - mid_val):
            return mid+1
    return mid


def merge_close_lines(ar, line_tol=2):
    """Merges lines which are within a tolerance by calculating a
    moving mean, based on their x or y axis projections.

    Parameters
    ----------
    ar : list
    line_tol : int, optional (default: 2)

    Returns
    -------
    ret : list

    """
    ret = []
    for a in ar:
        if not ret:
            ret.append(a)
        else:
            temp = ret[-1]
            if np.isclose(temp, a, atol=line_tol):
                temp = (temp + a) / 2.0
                ret[-1] = temp
            else:
                ret.append(a)
    return ret


def text_strip(text, strip=""):
    """Strips any characters in `strip` that are present in `text`.
    Parameters
    ----------
    text : str
        Text to process and strip.
    strip : str, optional (default: '')
        Characters that should be stripped from `text`.
    Returns
    -------
    stripped : str
    """
    if not strip:
        return text

    stripped = re.sub(
        fr"[{''.join(map(re.escape, strip))}]", "", text, re.UNICODE
    )
    return stripped


# TODO: combine the following functions into a TextProcessor class which
# applies corresponding transformations sequentially
# (inspired from sklearn.pipeline.Pipeline)


def flag_font_size(textline, direction, strip_text=""):
    """Flags super/subscripts in text by enclosing them with <s></s>.
    May give false positives.

    Parameters
    ----------
    textline : list
        List of PDFMiner LTChar objects.
    direction : string
        Direction of the PDFMiner LTTextLine object.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.

    Returns
    -------
    fstring : string

    """
    if direction == "horizontal":
        d = [
            (t.get_text(), np.round(t.height, decimals=6))
            for t in textline
            if not isinstance(t, LTAnno)
        ]
    elif direction == "vertical":
        d = [
            (t.get_text(), np.round(t.width, decimals=6))
            for t in textline
            if not isinstance(t, LTAnno)
        ]
    text_sizes = [np.round(size, decimals=6) for text, size in d]
    if len(set(text_sizes)) > 1:
        flist = []
        min_size = min(text_sizes)
        for key, chars in groupby(d, itemgetter(1)):
            if key == min_size:
                fchars = [t[0] for t in chars]
                if "".join(fchars).strip():
                    fchars.insert(0, "<s>")
                    fchars.append("</s>")
                    flist.append("".join(fchars))
            else:
                fchars = [t[0] for t in chars]
                if "".join(fchars).strip():
                    flist.append("".join(fchars))
        fstring = "".join(flist)
    else:
        fstring = "".join(t.get_text() for t in textline)
    return text_strip(fstring, strip_text)


def split_textline(table, textline, direction, flag_size=False, strip_text=""):
    """Split PDFMiner LTTextLine into substrings if it spans across
    multiple rows/columns.

    Parameters
    ----------
    table : camelot.core.Table
    textline : object
        PDFMiner LTTextLine object.
    direction : string
        Direction of the PDFMiner LTTextLine object.
    flag_size : bool, optional (default: False)
        Whether or not to highlight a substring using <s></s>
        if its size is different from rest of the string. (Useful for
        super and subscripts.)
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.

    Returns
    -------
    grouped_chars : list
        List of tuples of the form (idx, text) where idx is the index
        of row/column and text is the an lttextline substring.

    """
    cut_text = []
    bbox = textline.bbox
    try:
        if direction == "horizontal" and not textline.is_empty():
            x_overlap = [
                i
                for i, x in enumerate(table.cols)
                if x[0] <= bbox[2] and bbox[0] <= x[1]
            ]
            r_idx = [
                j
                for j, r in enumerate(table.rows)
                if r[1] <= (bbox[1] + bbox[3]) / 2 <= r[0]
            ]
            r = r_idx[0]
            x_cuts = [
                (c, table.cells[r][c].x2)
                for c in x_overlap
                if table.cells[r][c].right
            ]
            if not x_cuts:
                x_cuts = [(x_overlap[0], table.cells[r][-1].x2)]
            for obj in textline._objs:
                row = table.rows[r]
                for cut in x_cuts:
                    if isinstance(obj, LTChar):
                        if (
                            row[1] <= (obj.y0 + obj.y1) / 2 <= row[0]
                            and (obj.x0 + obj.x1) / 2 <= cut[1]
                        ):
                            cut_text.append((r, cut[0], obj))
                            break
                        # TODO: add test
                        if cut == x_cuts[-1]:
                            cut_text.append((r, cut[0] + 1, obj))
                    elif isinstance(obj, LTAnno):
                        cut_text.append((r, cut[0], obj))
        elif direction == "vertical" and not textline.is_empty():
            y_overlap = [
                j
                for j, y in enumerate(table.rows)
                if y[1] <= bbox[3] and bbox[1] <= y[0]
            ]
            c_idx = [
                i
                for i, c in enumerate(table.cols)
                if c[0] <= (bbox[0] + bbox[2]) / 2 <= c[1]
            ]
            c = c_idx[0]
            y_cuts = [
                (r, table.cells[r][c].y1)
                for r in y_overlap
                if table.cells[r][c].bottom
            ]
            if not y_cuts:
                y_cuts = [(y_overlap[0], table.cells[-1][c].y1)]
            for obj in textline._objs:
                col = table.cols[c]
                for cut in y_cuts:
                    if isinstance(obj, LTChar):
                        if col[0] <= (obj.x0 + obj.x1) / 2 <= col[1] \
                                and (obj.y0 + obj.y1) / 2 >= cut[1]:
                            cut_text.append((cut[0], c, obj))
                            break
                        # TODO: add test
                        if cut == y_cuts[-1]:
                            cut_text.append((cut[0] - 1, c, obj))
                    elif isinstance(obj, LTAnno):
                        cut_text.append((cut[0], c, obj))
    except IndexError:
        return [(-1, -1, textline.get_text())]
    grouped_chars = []
    for key, chars in groupby(cut_text, itemgetter(0, 1)):
        if flag_size:
            grouped_chars.append(
                (
                    key[0],
                    key[1],
                    flag_font_size(
                        [t[2] for t in chars], direction, strip_text=strip_text
                    ),
                )
            )
        else:
            gchars = [t[2].get_text() for t in chars]
            grouped_chars.append(
                (key[0], key[1], text_strip("".join(gchars), strip_text))
            )
    return grouped_chars


def get_table_index(
    table, t, direction, split_text=False, flag_size=False, strip_text=""
):
    """Gets indices of the table cell where given text object lies by
    comparing their y and x-coordinates.

    Parameters
    ----------
    table : camelot.core.Table
    t : object
        PDFMiner LTTextLine object.
    direction : string
        Direction of the PDFMiner LTTextLine object.
    split_text : bool, optional (default: False)
        Whether or not to split a text line if it spans across
        multiple cells.
    flag_size : bool, optional (default: False)
        Whether or not to highlight a substring using <s></s>
        if its size is different from rest of the string. (Useful for
        super and subscripts)
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.

    Returns
    -------
    indices : list
        List of tuples of the form (r_idx, c_idx, text) where r_idx
        and c_idx are row and column indices.
    error : float
        Assignment error, percentage of text area that lies outside
        a cell.
        +-------+
        |       |
        |   [Text bounding box]
        |       |
        +-------+

    """
    r_idx, c_idx = [-1] * 2
    for r in range(len(table.rows)):
        if (t.y0 + t.y1) / 2.0 < table.rows[r][0] and \
           (t.y0 + t.y1) / 2.0 > table.rows[r][1]:
            lt_col_overlap = []
            for c in table.cols:
                if c[0] <= t.x1 and c[1] >= t.x0:
                    left = t.x0 if c[0] <= t.x0 else c[0]
                    right = t.x1 if c[1] >= t.x1 else c[1]
                    lt_col_overlap.append(abs(left - right) / abs(c[0] - c[1]))
                else:
                    lt_col_overlap.append(-1)
            if len(list(filter(lambda x: x != -1, lt_col_overlap))) == 0:
                text = t.get_text().strip("\n")
                text_range = (t.x0, t.x1)
                col_range = (table.cols[0][0], table.cols[-1][1])
                warnings.warn(
                    f"{text} {text_range} does not lie in column range "
                    f"{col_range}"
                )
            r_idx = r
            c_idx = lt_col_overlap.index(max(lt_col_overlap))
            break

    # error calculation
    y0_offset, y1_offset, x0_offset, x1_offset = [0] * 4
    if t.y0 > table.rows[r_idx][0]:
        y0_offset = abs(t.y0 - table.rows[r_idx][0])
    if t.y1 < table.rows[r_idx][1]:
        y1_offset = abs(t.y1 - table.rows[r_idx][1])
    if t.x0 < table.cols[c_idx][0]:
        x0_offset = abs(t.x0 - table.cols[c_idx][0])
    if t.x1 > table.cols[c_idx][1]:
        x1_offset = abs(t.x1 - table.cols[c_idx][1])
    X = 1.0 if abs(t.x0 - t.x1) == 0.0 else abs(t.x0 - t.x1)
    Y = 1.0 if abs(t.y0 - t.y1) == 0.0 else abs(t.y0 - t.y1)
    charea = X * Y
    error = (
        (X * (y0_offset + y1_offset)) + (Y * (x0_offset + x1_offset))
    ) / charea

    if split_text:
        return (
            split_textline(
                table, t, direction, flag_size=flag_size, strip_text=strip_text
            ),
            error,
        )
    if flag_size:
        return (
            [
                (
                    r_idx,
                    c_idx,
                    flag_font_size(t._objs,
                                   direction,
                                   strip_text=strip_text),
                )
            ],
            error,
        )
    return [(r_idx, c_idx, text_strip(t.get_text(), strip_text))], \
        error


def compute_accuracy(error_weights):
    """Calculates a score based on weights assigned to various
    parameters and their error percentages.

    Parameters
    ----------
    error_weights : list
        Two-dimensional list of the form [[p1, e1], [p2, e2], ...]
        where pn is the weight assigned to list of errors en.
        Sum of pn should be equal to 100.

    Returns
    -------
    score : float

    """
    SCORE_VAL = 100
    try:
        score = 0
        if sum(ew[0] for ew in error_weights) != SCORE_VAL:
            raise ValueError("Sum of weights should be equal to 100.")
        for ew in error_weights:
            weight = ew[0] / len(ew[1])
            for error_percentage in ew[1]:
                score += weight * (1 - error_percentage)
    except ZeroDivisionError:
        score = 0
    return score


def compute_whitespace(d):
    """Calculates the percentage of empty strings in a
    two-dimensional list.

    Parameters
    ----------
    d : list

    Returns
    -------
    whitespace : float
        Percentage of empty cells.

    """
    whitespace = 0
    for i in d:
        for j in i:
            if j.strip() == "":
                whitespace += 1
    whitespace = 100 * (whitespace / float(len(d) * len(d[0])))
    return whitespace


def get_page_layout(
        filename,
        char_margin=1.0,
        line_margin=0.5,
        word_margin=0.1,
        detect_vertical=True,
        all_texts=True):
    """Returns a PDFMiner LTPage object and page dimension of a single
    page pdf. See https://euske.github.io/pdfminer/ to get definitions
    of kwargs.

    Parameters
    ----------
    filename : string
        Path to pdf file.
    char_margin : float
    line_margin : float
    word_margin : float
    detect_vertical : bool
    all_texts : bool

    Returns
    -------
    layout : object
        PDFMiner LTPage object.
    dim : tuple
        Dimension of pdf page in the form (width, height).

    """
    with open(filename, "rb") as f:
        parser = PDFParser(f)
        document = PDFDocument(parser)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        laparams = LAParams(
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            width = layout.bbox[2]
            height = layout.bbox[3]
            dim = (width, height)
            break  # we assume a single page pdf
        return layout, dim


def get_text_objects(layout, ltype="char", t=None):
    """Recursively parses pdf layout to get a list of
    PDFMiner text objects.

    Parameters
    ----------
    layout : object
        PDFMiner LTPage object.
    ltype : string
        Specify 'char', 'lh', 'lv' to get LTChar, LTTextLineHorizontal,
        and LTTextLineVertical objects respectively.
    t : list

    Returns
    -------
    t : list
        List of PDFMiner text objects.

    """
    if ltype == "char":
        LTObject = LTChar
    elif ltype == "image":
        LTObject = LTImage
    elif ltype == "horizontal_text":
        LTObject = LTTextLineHorizontal
    elif ltype == "vertical_text":
        LTObject = LTTextLineVertical
    if t is None:
        t = []
    try:
        for obj in layout._objs:
            if isinstance(obj, LTObject):
                t.append(obj)
            else:
                t += get_text_objects(obj, ltype=ltype)
    except AttributeError:
        pass
    return t


def export_pdf_as_png(pdf_path, destination_path, resolution=300):
    """Generate an image from a pdf.

    Parameters
    ----------
    pdf_path : str
    destination_path : str
    resolution : int
    """
    gs_call = "-q -sDEVICE=png16m -o " \
        "{destination_path} -r{resolution} {pdf_path}" \
        .format(
            destination_path=destination_path,
            resolution=resolution,
            pdf_path=pdf_path
        )
    gs_call = gs_call.encode().split()
    null = open(os.devnull, "wb")
    Ghostscript(*gs_call, stdout=null)
    null.close()


def compare_tables(left, right):
    """Compare two tables and displays differences in a human readable form.

    Parameters
    ----------
    left : data frame
    right : data frame
    """
    diff_cols = right.shape[1]-left.shape[1]
    diff_rows = right.shape[0]-left.shape[0]
    differences = []
    if diff_rows:
        differences.append(
            "{diff_rows} {more_fewer} rows".format(
                diff_rows=abs(diff_rows),
                more_fewer='more' if diff_rows > 0 else 'fewer'
            )
        )
    if diff_cols:
        differences.append(
            "{diff_cols} {more_fewer} columns".format(
                diff_cols=abs(diff_cols),
                more_fewer='more' if diff_cols > 0 else 'fewer'
            )
        )
    if differences:
        differences_str = " and ".join(differences)
        print(
            "Right has {differences_str} than left "
            "{shape_left} vs {shape_right}".format(
                differences_str=differences_str,
                shape_left=[left.shape[0], left.shape[1]],
                shape_right=[right.shape[0], right.shape[1]],
            )
        )

    table1, table2 = [left, right]
    name_table1, name_table2 = ["left", "right"]
    if not diff_cols:
        # Same number of cols: compare rows since they're of the same length
        if diff_rows > 0:
            # Use the longest table as a reference
            table1, table2 = table2, table1
            name_table1, name_table2 = name_table2, name_table1
        for index, lrow in table1.iterrows():
            if index < table2.shape[0]:
                srow = table2.loc[index, :]
                if not lrow.equals(srow):
                    diff_df = pd.DataFrame()
                    diff_df = diff_df.append(lrow, ignore_index=True)
                    diff_df = diff_df.append(srow, ignore_index=True)
                    diff_df.insert(0, 'Table', [name_table1, name_table2])
                    print("Row {index} differs:".format(index=index))
                    print(diff_df.values)
                    break
            else:
                print("Row {index} unique to {name_table1}: {lrow}".format(
                    index=index,
                    name_table1=name_table1,
                    lrow=lrow
                ))
                break
    elif not diff_rows:
        # Same number of rows: compare columns since they're of the same length
        if diff_cols > 0:
            # Use the longest table as a reference
            table1, table2 = table2, table1
            name_table1, name_table2 = name_table2, name_table1
        for i, col in enumerate(table1.columns):
            lcol = table1.iloc[:, i]
            if col in table2:
                scol = table2.iloc[:, i]
                if not lcol.equals(scol):
                    diff_df = pd.DataFrame()
                    diff_df[name_table1] = scol
                    diff_df[name_table2] = lcol
                    diff_df["Match"] = lcol == scol
                    print(
                        "Column {i} different:\n"
                        "{diff_df}".format(
                            i=i,
                            diff_df=diff_df
                        )
                    )
                    break
            else:
                print("Column {i} unique to {name_table1}: {lcol}")
                break
    else:
        print("Tables have different shapes")
