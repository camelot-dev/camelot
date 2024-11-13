"""General helper utilities to parse the pdf tables."""

from __future__ import annotations

import atexit
import math
import os
import random
import re
import shutil
import string
import tempfile
import warnings
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any
from typing import Callable
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc
from urllib.parse import uses_params
from urllib.parse import uses_relative
from urllib.request import Request
from urllib.request import urlopen

import numpy as np
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.layout import LTAnno
from pdfminer.layout import LTChar
from pdfminer.layout import LTContainer
from pdfminer.layout import LTImage
from pdfminer.layout import LTItem
from pdfminer.layout import LTTextLine
from pdfminer.layout import LTTextLineHorizontal
from pdfminer.layout import LTTextLineVertical
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from pypdf._utils import StrByteType


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
    """Generate a random string .

    Parameters
    ----------
    length : int
        The length of the string to return.

    Returns
    -------
    string
        returns a random string
    """
    ret = ""
    while length:
        ret += random.choice(  # noqa S311
            string.digits + string.ascii_lowercase + string.ascii_uppercase
        )
        length -= 1
    return ret


def download_url(url: str) -> StrByteType | Path:
    """Download file from specified URL.

    Parameters
    ----------
    url : str

    Returns
    -------
    filepath : Union[StrByteType, Path]
        Temporary filepath.

    """
    filename = f"{random_string(6)}.pdf"
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:  # noqa S310
        # Valid url checking has been done in function is_url
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip;q=1.0, deflate;q=0.9, br;q=0.8, compress;q=0.7, *;q=0.1",
        }
        request = Request(url, None, headers)
        obj = urlopen(request)  # noqa S310
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
    "table_regions",
    "backend",
]
text_kwargs = common_kwargs + ["columns", "edge_tol", "row_tol", "column_tol"]
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
    "use_fallback",
]
flavor_to_kwargs = {
    "stream": text_kwargs,
    "network": text_kwargs,
    "lattice": lattice_kwargs,
    "hybrid": text_kwargs + lattice_kwargs,
}


def validate_input(kwargs, flavor="lattice"):
    """Validates input keyword arguments.

    Parameters
    ----------
    kwargs : [type]
        [description]
    flavor : str, optional
        [description], by default "lattice"

    Raises
    ------
    ValueError
        [description]
    """
    parser_kwargs = flavor_to_kwargs[flavor]
    # s.difference(t): new set with elements in s but not in t
    isec = set(kwargs.keys()).difference(set(parser_kwargs))
    if isec:
        raise ValueError(
            "{} cannot be used with flavor='{}'".format(",".join(sorted(isec)), flavor)
        )


def remove_extra(kwargs, flavor="lattice"):
    """Remove extra key - value pairs from a kwargs dictionary.

    Parameters
    ----------
    kwargs : [type]
        [description]
    flavor : str, optional
        [description], by default "lattice"

    Returns
    -------
    [type]
        [description]

    """
    parser_kwargs = flavor_to_kwargs[flavor]
    # Avoid "dictionary changed size during iteration"
    kwargs_keys = list(kwargs.keys())
    for key in kwargs_keys:
        if key not in parser_kwargs:
            kwargs.pop(key)
    return kwargs


# https://stackoverflow.com/a/22726782
# and https://stackoverflow.com/questions/10965479
class TemporaryDirectory:
    """A class method that will be used to create temporary directories."""

    def __enter__(self):
        """Enter the temporary directory .

        Returns
        -------
        [type]
            [description]
        """
        self.name = tempfile.mkdtemp()
        # Only delete the temporary directory upon
        # program exit.
        atexit.register(shutil.rmtree, self.name)
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        """Called when the client exits.

        Parameters
        ----------
        exc_type : [type]
            [description]
        exc_value : [type]
            [description]
        traceback : [type]
            [description]
        """
        pass


def build_file_path_in_temp_dir(filename, extension=None):
    """Generate a new path within a temporary directory.

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
        path = os.path.join(temp_dir, filename)
    return path


def translate(x1: float, x2: float) -> float:
    """Translate x2 by x1.

    Parameters
    ----------
    x1 : float
        The offset to apply.
    x2 : float
        The original y-coordinate.

    Returns
    -------
    float
        The translated y-coordinate.

    """
    return x2 + x1


def scale(value: float, factor: float) -> float:
    """Scale a given value by a factor.

    Parameters
    ----------
    value : float
        The value to scale.
    factor : float
        The scaling factor.

    Returns
    -------
    float
        The scaled value.
    """
    return value * factor


def scale_pdf(k, factors):
    """Translate and scale pdf coordinate space to image coordinate space.

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


def scale_image(
    tables: dict[tuple[float, float, float, float], list[tuple[float, float]]],
    v_segments: list[tuple[float, float, float, float]],
    h_segments: list[tuple[float, float, float, float]],
    factors: tuple[float, float, float],
) -> tuple[
    dict[tuple[float, float, float, float], dict[str, list[tuple[float, float]]]],
    list[tuple[float, float, float, float]],
    list[tuple[float, float, float, float]],
]:
    """Translate and scale image coordinate space to PDF coordinate space.

    Parameters
    ----------
    tables : dict
        A dictionary with table boundaries as keys (tuples of four floats)
        and a list of intersections (list of tuples of two floats) as values.
    v_segments : list
        A list of vertical line segments, where each segment is a tuple
        of four floats (x1, y1, x2, y2).
    h_segments : list
        A list of horizontal line segments, where each segment is a tuple
        of four floats (x1, y1, x2, y2).
    factors : tuple
        A tuple (scaling_factor_x, scaling_factor_y, img_y) where the
        first two elements are scaling factors and img_y is the height of
        the image.

    Returns
    -------
    Tuple[Dict[Tuple[float, float, float, float], Dict[str, List[Tuple[float, float]]]],
          List[Tuple[float, float, float, float]],
          List[Tuple[float, float, float, float]]]
        A tuple containing:
        - tables_new: A new dictionary with scaled table boundaries and joints.
        - v_segments_new: A new list of scaled vertical segments.
        - h_segments_new: A new list of scaled horizontal segments.
    """
    scaling_factor_x, scaling_factor_y, img_y = factors
    tables_new = {}

    for k in tables.keys():
        x1, y1, x2, y2 = k
        x1 = scale(x1, scaling_factor_x)
        y1 = scale(abs(translate(-img_y, y1)), scaling_factor_y)
        x2 = scale(x2, scaling_factor_x)
        y2 = scale(abs(translate(-img_y, y2)), scaling_factor_y)

        # j_x and j_y are tuples of floats
        j_x, j_y = zip(*tables[k])  # noqa B905
        j_x_scaled = [scale(j, scaling_factor_x) for j in j_x]
        j_y_scaled = [scale(abs(translate(-img_y, j)), scaling_factor_y) for j in j_y]

        tables_new[(x1, y1, x2, y2)] = {
            "joints": list(zip(j_x_scaled, j_y_scaled))  # noqa B905
        }

    # Scale vertical segments
    v_segments_new = []
    for v in v_segments:
        x1, x2 = scale(v[0], scaling_factor_x), scale(v[2], scaling_factor_x)
        y1, y2 = (
            scale(abs(translate(-img_y, v[1])), scaling_factor_y),
            scale(abs(translate(-img_y, v[3])), scaling_factor_y),
        )
        v_segments_new.append((x1, y1, x2, y2))

    # Scale horizontal segments
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
    """Get text rotation.

    Detects if text in table is rotated or not using the current
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
        clockwise = sum(t.matrix[1] < 0 and t.matrix[2] > 0 for t in chars)
        anticlockwise = sum(t.matrix[1] > 0 and t.matrix[2] < 0 for t in chars)
        rotation = "anticlockwise" if clockwise < anticlockwise else "clockwise"
    return rotation


def segments_in_bbox(bbox, v_segments, h_segments):
    """Return all line segments present inside a bounding box.

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
        if v[1] > lb[1] - 2 and v[3] < rt[1] + 2 and lb[0] - 2 <= v[0] <= rt[0] + 2
    ]
    h_s = [
        h
        for h in h_segments
        if h[0] > lb[0] - 2 and h[2] < rt[0] + 2 and lb[1] - 2 <= h[1] <= rt[1] + 2
    ]
    return v_s, h_s


def get_textline_coords(textline):
    """Calculate the coordinates of each alignment for a given textline."""
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
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def bboxes_overlap(bbox1, bbox2):
    """Check if boundingboxes overlap.

    Parameters
    ----------
    bbox1 : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.
    bbox2 : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
        space.

    Returns
    -------
    bool
        Returns True if two bounding boxes overlap
    """
    (left1, bottom1, right1, top1) = bbox1
    (left2, bottom2, right2, top2) = bbox2
    return ((left1 < left2 < right1) or (left1 < right2 < right1)) and (
        (bottom1 < bottom2 < top1) or (bottom1 < top2 < top1)
    )


def textlines_overlapping_bbox(bbox, textlines):
    """Return all text objects which overlap or are within a bounding box.

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
    t_bbox = [t for t in textlines if bboxes_overlap(bbox, (t.x0, t.y0, t.x1, t.y1))]
    return t_bbox


def text_in_bbox(bbox, text):
    """Return all text objects in a bounding box.

    Return the text objects which lie at least 80% inside a bounding box
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
        List of PDFMiner text objects that lie inside table, discarding the overlapping ones

    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    t_bbox = [
        t
        for t in text
        if lb[0] - 2 <= (t.x0 + t.x1) / 2.0 <= rt[0] + 2
        and lb[1] - 2 <= (t.y0 + t.y1) / 2.0 <= rt[1] + 2
    ]

    # Avoid duplicate text by discarding overlapping boxes
    rest = {t for t in t_bbox}
    for ba in t_bbox:
        for bb in rest.copy():
            if ba == bb:
                continue
            if bbox_intersect(ba, bb):
                ba_area = bbox_area(ba)
                # if the intersection is larger than 80% of ba's size, we keep the longest
                if ba_area == 0 or (bbox_intersection_area(ba, bb) / ba_area) > 0.8:
                    if bbox_longer(bb, ba):
                        rest.discard(ba)
    unique_boxes = list(rest)

    return unique_boxes


def text_in_bbox_per_axis(bbox, horizontal_text, vertical_text):
    """Return all text objects present inside a bounding box.

    split between horizontal and vertical text.

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
    """Expand (if needed) a bbox so that it fits the parameter textline."""
    return (
        min(bbox[0], textline.x0),
        min(bbox[1], textline.y0),
        max(bbox[2], textline.x1),
        max(bbox[3], textline.y1),
    )


def bbox_from_textlines(textlines):
    """Return the smallest bbox containing all the text objects passed as a parameters.

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
    bbox = (textlines[0].x0, textlines[0].y0, textlines[0].x1, textlines[0].y1)

    for tl in textlines[1:]:
        bbox = expand_bbox_with_textline(bbox, tl)
    return bbox


def bbox_intersection_area(ba, bb) -> float:
    """Return area of the intersection of the bounding boxes of two PDFMiner objects.

    Parameters
    ----------
    ba : PDFMiner text object
    bb : PDFMiner text object

    Returns
    -------
    intersection_area : float
        Area of the intersection of the bounding boxes of both objects

    """
    x_left = max(ba.x0, bb.x0)
    y_top = min(ba.y1, bb.y1)
    x_right = min(ba.x1, bb.x1)
    y_bottom = max(ba.y0, bb.y0)

    if x_right < x_left or y_bottom > y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_top - y_bottom)
    return intersection_area


def bbox_area(bb) -> float:
    """Return area of the bounding box of a PDFMiner object.

    Parameters
    ----------
    bb : PDFMiner text object

    Returns
    -------
    area : float
        Area of the bounding box of the object

    """
    return (bb.x1 - bb.x0) * (bb.y1 - bb.y0)


def bbox_intersect(ba, bb) -> bool:
    """Return True if the bounding boxes of two PDFMiner objects intersect.

    Parameters
    ----------
    ba : PDFMiner text object
    bb : PDFMiner text object

    Returns
    -------
    overlaps : bool
        True if the bounding boxes intersect

    """
    return ba.x1 >= bb.x0 and bb.x1 >= ba.x0 and ba.y1 >= bb.y0 and bb.y1 >= ba.y0


def find_columns_boundaries(tls, min_gap=1.0):
    """Make a list of disjunct cols boundaries for a list of text objects.

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
    # Check if boundaries list is empty
    if not boundaries:
        return []

    # Check if boundaries have at least one tuple
    if len(boundaries) < 1:
        return []

    # From the row boundaries, identify splits by getting the mid points
    # between the boundaries.
    anchors = list(
        map(
            lambda idx: (boundaries[idx - 1][1] + boundaries[idx][0]) / 2.0,
            range(1, len(boundaries)),
        )
    )

    # Insert the first boundary's left coordinate
    anchors.insert(0, boundaries[0][0])

    # Append the last boundary's right coordinate
    anchors.append(boundaries[-1][1])

    return anchors


def get_index_closest_point(
    point: Any, sorted_list: list[Any], fn: Callable[[Any], Any] = lambda x: x
) -> int | None:
    """Find the index of the closest point in sorted_list.

    Parameters
    ----------
    point : Any
        The reference sortable element to search.
    sorted_list : List[Any]
        A sorted list of elements.
    fn : Callable[[Any], Any], optional
        Optional accessor function, by default lambda x: x

    Returns
    -------
    Optional[int]
        The index of the closest point, or None if the list is empty.
    """
    n = len(sorted_list)

    # If the list is empty, return None
    if n == 0:
        return None

    # Edge cases for points outside the range of the sorted list
    if point <= fn(sorted_list[0]):
        return 0
    if point >= fn(sorted_list[-1]):
        return n - 1

    # Binary search
    left, right = 0, n - 1

    while left < right:
        mid = (left + right) // 2
        mid_val = fn(sorted_list[mid])

        if mid_val < point:
            left = mid + 1
        else:
            right = mid

    # After the loop, left is the first index greater than or equal to the point
    # We need to check which of the closest points is closer to the reference point
    if left == 0:
        return 0
    if left == n:
        return n - 1

    # Compare the closest two points
    if abs(fn(sorted_list[left]) - point) < abs(fn(sorted_list[left - 1]) - point):
        return left
    else:
        return left - 1


def bbox_longer(ba, bb) -> bool:
    """Return True if the bounding box of the first PDFMiner object is longer or equal to the second.

    Parameters
    ----------
    ba : PDFMiner text object
    bb : PDFMiner text object

    Returns
    -------
    longer : bool
        True if the bounding box of the first object is longer or equal

    """
    return (ba.x1 - ba.x0) >= (bb.x1 - bb.x0)


def merge_close_lines(ar, line_tol=2):
    """Merge lines which are within a tolerance.

    By calculating a moving mean, based on their x or y axis projections.

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
            if math.isclose(temp, a, abs_tol=line_tol):
                temp = (temp + a) / 2.0
                ret[-1] = temp
            else:
                ret.append(a)
    return ret


def text_strip(text, strip=""):
    """Strip any characters in `strip` that are present in `text`.

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
        rf"[{''.join(map(re.escape, strip))}]", "", text, flags=re.UNICODE
    )
    return stripped


# TODO: combine the following functions into a TextProcessor class which
# applies corresponding transformations sequentially
# (inspired from sklearn.pipeline.Pipeline)


def flag_font_size(
    textline: list[LTChar | LTAnno], direction: str, strip_text: str = ""
) -> str:
    """Flag super/subscripts.

    Flag super/subscripts in text by enclosing them with <s></s>.
    May give false positives.

    Parameters
    ----------
    textline : List[LTChar | LTAnno]
        List of objects implementing the LTCharProtocol.
    direction : str
        Direction of the PDFMiner LTTextLine object.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.

    Returns
    -------
    str
        The processed string with flagged super/subscripts.
    """
    # Determine size based on direction and collect text and size
    d: list[tuple[str, float]] = []
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
    else:
        raise ValueError("Invalid direction provided. Use 'horizontal' or 'vertical'.")

    # Group characters by size
    size_groups: dict[float, list[str]] = {}
    for text, size in d:
        size_groups.setdefault(size, []).append(text)

    # Check if we have multiple font sizes
    if len(size_groups) > 1:
        min_size = min(size_groups.keys())
        flist: list[str] = []

        for size, chars in size_groups.items():
            combined_chars = "".join(chars).strip()
            if combined_chars:
                if size == min_size:
                    flist.append(f"<s>{combined_chars}</s>")
                else:
                    flist.append(combined_chars)

        fstring = "".join(flist)
    else:
        fstring = "".join(text for text, _ in d)

    return text_strip(fstring, strip_text)


def split_textline(
    table: Any,
    textline: LTTextLine,
    direction: str,
    flag_size: bool = False,
    strip_text: str = "",
) -> list[tuple[int, int, str]]:
    """Split textline into substrings if it spans across multiple rows/columns.

    Parameters
    ----------
    table : camelot.core.Table
        The table structure containing rows and columns.
    textline : LTTextLine
        PDFMiner LTTextLine object.
    direction : str
        Direction of the PDFMiner LTTextLine object, either "horizontal" or "vertical".
    flag_size : bool
        Whether to highlight a substring using <s></s> if its size differs from the rest of the string.
    strip_text : str
        Characters to strip from a string before assigning it to a cell.

    Returns
    -------
    List[tuple[int, int, str]]
        A list of tuples of the form (idx, text) where idx is the index of row/column
        and text is an LTTextLine substring.
    """
    cut_text: list[tuple[int, int, LTChar | LTAnno]] = []
    bbox = textline.bbox

    if textline.is_empty():
        return [(-1, -1, textline.get_text())]

    if direction == "horizontal":
        cut_text = _process_horizontal_cut(table, textline, bbox)
    elif direction == "vertical":
        cut_text = _process_vertical_cut(table, textline, bbox)

    grouped_chars = _group_and_process_chars(cut_text, flag_size, direction, strip_text)
    return grouped_chars


def _process_horizontal_cut(
    table, textline, bbox
) -> list[tuple[int, int, LTChar | LTAnno]]:
    """Process horizontal cuts of the textline."""
    cut_text: list[tuple[int, int, LTChar | LTAnno]] = []
    x_overlap = [
        i for i, x in enumerate(table.cols) if x[0] <= bbox[2] and bbox[0] <= x[1]
    ]
    r_idx = [
        j for j, r in enumerate(table.rows) if r[1] <= (bbox[1] + bbox[3]) / 2 <= r[0]
    ]

    if not r_idx:
        return cut_text

    r = r_idx[0]
    x_cuts = [
        (c, table.cells[r][c].x2) for c in x_overlap if table.cells[r][c].right
    ] or [(x_overlap[0], table.cells[r][-1].x2)]

    for obj in textline._objs:
        row = table.rows[r]
        for cut in x_cuts:
            if (
                isinstance(obj, LTChar)
                and row[1] <= (obj.y0 + obj.y1) / 2 <= row[0]
                and (obj.x0 + obj.x1) / 2 <= cut[1]
            ):
                cut_text.append((r, cut[0], obj))
                break
            elif isinstance(obj, LTAnno):
                cut_text.append((r, cut[0], obj))
    return cut_text


def _process_vertical_cut(
    table, textline, bbox
) -> list[tuple[int, int, LTChar | LTAnno]]:
    """Process vertical cuts of the textline."""
    cut_text: list[tuple[int, int, LTChar | LTAnno]] = []
    y_overlap = [
        j for j, y in enumerate(table.rows) if y[1] <= bbox[3] and bbox[1] <= y[0]
    ]
    c_idx = [
        i for i, c in enumerate(table.cols) if c[0] <= (bbox[0] + bbox[2]) / 2 <= c[1]
    ]

    if not c_idx:
        return cut_text

    c = c_idx[0]
    y_cuts = [
        (r, table.cells[r][c].y1) for r in y_overlap if table.cells[r][c].bottom
    ] or [(y_overlap[0], table.cells[-1][c].y1)]

    for obj in textline._objs:
        col = table.cols[c]
        for cut in y_cuts:
            if (
                isinstance(obj, LTChar)
                and col[0] <= (obj.x0 + obj.x1) / 2 <= col[1]
                and (obj.y0 + obj.y1) / 2 >= cut[1]
            ):
                cut_text.append((cut[0], c, obj))
                break
            elif isinstance(obj, LTAnno):
                cut_text.append((cut[0], c, obj))
    return cut_text


def _group_and_process_chars(
    cut_text: list[tuple[int, int, LTChar | LTAnno]],
    flag_size: bool,
    direction: str,
    strip_text: str,
) -> list[tuple[int, int, str]]:
    """
    Group characters and process them based on size flag.

    Parameters
    ----------
    cut_text : list of tuples
        Each tuple consists of (x0, y0, character), where x0 and y0 are
        coordinates and character can be an instance of LTChar or LTAnno.

    flag_size : bool
        A flag indicating whether to group by font size.

    direction : str
        Direction for processing the text (e.g., 'horizontal' or 'vertical').

    strip_text : str
        Characters to strip from the text.

    Returns
    -------
    list of tuples
        Each tuple consists of (x0, y0, processed_text), where processed_text
        is the grouped and processed text based on the specified conditions.
    """
    grouped_chars: list[tuple[int, int, str]] = []

    for key, chars in groupby(cut_text, itemgetter(0, 1)):
        chars_list = list(chars)  # Convert the iterator to a list to reuse it

        if flag_size:
            grouped_chars.append(
                (
                    key[0],
                    key[1],
                    flag_font_size(
                        [t[2] for t in chars_list], direction, strip_text=strip_text
                    ),
                )
            )
        else:
            gchars = []
            for t in chars_list:
                gchars.append(t[2].get_text())

            grouped_chars.append(
                (key[0], key[1], text_strip("".join(gchars), strip_text))
            )

    return grouped_chars


def get_table_index(
    table, t, direction, split_text=False, flag_size=False, strip_text=""
):
    """
    Get indices of the table cell.

    Get the index of a table cell where a given text object lies by
    comparing their y and x-coordinates.

    Parameters
    ----------
    table : camelot.core.Table
        The table structure containing rows and columns.
    t : object
        PDFMiner LTTextLine object.
    direction : string
        Direction of the PDFMiner LTTextLine object.
    split_text : bool, optional (default: False)
        Whether or not to split a text line if it spans across multiple cells.
    flag_size : bool, optional (default: False)
        Whether to highlight a substring using <s></s> if its size is different
        from the rest of the string.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before assigning it to a cell.

    Returns
    -------
    list
        List of tuples of the form (r_idx, c_idx, text) where r_idx and c_idx
        are row and column indices, respectively.
    float
        Assignment error, percentage of text area that lies outside a cell.
        +-------+
        |       |
        |   [Text bounding box]
        |       |
        +-------+
    """
    r_idx, c_idx = [-1] * 2
    for r in range(len(table.rows)):  # noqa
        if (t.y0 + t.y1) / 2.0 < table.rows[r][0] and (t.y0 + t.y1) / 2.0 > table.rows[
            r
        ][1]:
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
                    f"{text} {text_range} does not lie in column range {col_range}",
                    stacklevel=1,
                )
            r_idx = r
            c_idx = lt_col_overlap.index(max(lt_col_overlap))
            break
    if r_idx == -1:
        return [], 1.0  # Return early if no valid row is found

    error = calculate_assignment_error(t, table, r_idx, c_idx)

    if split_text:
        return (
            split_textline(
                table, t, direction, flag_size=flag_size, strip_text=strip_text
            ),
            error,
        )
        text = t.get_text().strip("\n")
    if flag_size:
        return [
            (r_idx, c_idx, flag_font_size(t._objs, direction, strip_text=strip_text))
        ], error
    else:
        return [(r_idx, c_idx, text_strip(t.get_text(), strip_text))], error


def calculate_assignment_error(t, table, r_idx, c_idx):
    """
    Calculate the assignment error for the given text object.

    Parameters
    ----------
    t : object
        PDFMiner LTTextLine object.
    table : camelot.core.Table
        The table structure containing rows and columns.
    r_idx : int
        Row index where the text object is located.
    c_idx : int
        Column index where the text object is located.

    Returns
    -------
    float
        The calculated assignment error.
    """
    y0_offset, y1_offset, x0_offset, x1_offset = [0] * 4
    if t.y0 > table.rows[r_idx][0]:
        y0_offset = abs(t.y0 - table.rows[r_idx][0])
    if t.y1 < table.rows[r_idx][1]:
        y1_offset = abs(t.y1 - table.rows[r_idx][1])
    if t.x0 < table.cols[c_idx][0]:
        x0_offset = abs(t.x0 - table.cols[c_idx][0])
    if t.x1 > table.cols[c_idx][1]:
        x1_offset = abs(t.x1 - table.cols[c_idx][1])

    x = 1.0 if abs(t.x0 - t.x1) == 0.0 else abs(t.x0 - t.x1)
    y = 1.0 if abs(t.y0 - t.y1) == 0.0 else abs(t.y0 - t.y1)

    charea = x * y
    error = ((x * (y0_offset + y1_offset)) + (y * (x0_offset + x1_offset))) / charea
    return error


def compute_accuracy(error_weights):
    """Compute Accuracy.

    Calculates a score based on weights assigned to various
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
    score_val = 100
    try:
        score = 0
        if sum([ew[0] for ew in error_weights]) != score_val:
            raise ValueError("Sum of weights should be equal to 100.")
        for ew in error_weights:
            weight = ew[0] / len(ew[1])
            for error_percentage in ew[1]:
                score += weight * (1 - error_percentage)
    except ZeroDivisionError:
        score = 0
    return score


def compute_whitespace(d: list[list[str]]) -> float:
    """Calculates the percentage of empty strings in a two-dimensional list.

    Parameters
    ----------
    d : list
        A two-dimensional list (list of lists) containing strings.

    Returns
    -------
    whitespace : float
        Percentage of empty cells.
    """
    # Initialize the count of empty strings
    whitespace = 0
    total_elements = 0  # Keep track of the total number of elements

    # Iterate through each row in the 2D list
    for i in d:
        # Only process if the row is a list
        if isinstance(i, list):
            total_elements += len(i)  # Count the number of elements in this row
            # Iterate through each element in the row
            for j in i:
                # Check if the element is an empty string after stripping whitespace
                if isinstance(j, str) and j.strip() == "":
                    whitespace += 1  # Increment the count of empty strings

    # Avoid division by zero
    if total_elements == 0:
        return 0.0  # If there are no elements, return 0%

    # Calculate the percentage of empty strings
    whitespace_percentage = 100 * (whitespace / total_elements)

    return whitespace_percentage


def get_page_layout(
    filename,
    line_overlap=0.5,
    char_margin=1.0,
    line_margin=0.5,
    word_margin=0.1,
    boxes_flow=0.5,
    detect_vertical=True,
    all_texts=True,
):
    """Return a PDFMiner LTPage object and page dimension of a single page pdf.

    To get the definitions of kwargs, see
    https://pdfminersix.rtfd.io/en/latest/reference/composable.html.

    Parameters
    ----------
    filename : string
        Path to pdf file.
    line_overlap : float
    char_margin : float
    line_margin : float
    word_margin : float
    boxes_flow : float
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
            raise PDFTextExtractionNotAllowed(
                f"Text extraction is not allowed: {filename}"
            )
        laparams = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        page = next(PDFPage.create_pages(document), None)
        if page is None:
            raise PDFTextExtractionNotAllowed
        interpreter.process_page(page)
        layout = device.get_result()
        width = layout.bbox[2]
        height = layout.bbox[3]
        dim = (width, height)
        return layout, dim


def get_char_objects(layout: LTContainer[Any]) -> list[LTChar]:
    """Get charachter objects from a pdf layout.

    Recursively parses pdf layout to get a list of PDFMiner LTChar

    Parameters
    ----------
    layout : object
        PDFMiner LTContainer object.

    Returns
    -------
    result : list
        List of LTChar text objects.

    """
    char = []
    try:
        for _object in layout:
            if isinstance(_object, LTChar):
                char.append(_object)
            elif isinstance(_object, LTContainer):
                child_char = get_char_objects(_object)
                char.extend(child_char)
    except AttributeError:
        pass
    return char


def get_image_char_and_text_objects(
    layout: LTContainer[LTItem],
) -> tuple[
    list[LTImage], list[LTChar], list[LTTextLineHorizontal], list[LTTextLineVertical]
]:
    """Parse a PDF layout to get objects.

    Recursively parses pdf layout to get a list of
    PDFMiner LTImage, LTTextLineHorizontal, LTTextLineVertical objects.

    Parameters
    ----------
    layout : object
        PDFMiner LTContainer object
            ( LTPage, LTTextLineHorizontal, LTTextLineVertical).

    Returns
    -------
    result : tuple
        Include List of LTImage objects, list of LTTextLineHorizontal objects
        and list of LTTextLineVertical objects

    """
    image = []
    char = []
    horizontal_text = []
    vertical_text = []

    try:
        for _object in layout:
            if isinstance(_object, LTImage):
                image.append(_object)
            elif isinstance(_object, LTTextLineHorizontal):
                horizontal_text.append(_object)
            elif isinstance(_object, LTTextLineVertical):
                vertical_text.append(_object)
            if isinstance(_object, LTChar):
                char.append(_object)
            elif isinstance(_object, LTContainer):
                child_image, child_char, child_horizontal_text, child_vertical_text = (
                    get_image_char_and_text_objects(_object)
                )
                image.extend(child_image)
                child_char = get_char_objects(_object)
                char.extend(child_char)
                horizontal_text.extend(child_horizontal_text)
                vertical_text.extend(child_vertical_text)
    except AttributeError:
        pass
    return image, char, horizontal_text, vertical_text
