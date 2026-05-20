"""Functions useful for detecting graphical elements from the image to using OpenCV to reconstruct / detect tables."""

import cv2
import numpy as np
from playa.miner import LTContainer
from playa.miner import LTLine
from playa.miner import LTRect

#: Minimum contour area, expressed as a fraction of the page image area,
#: for a contour to be considered a candidate table. The previous code
#: capped the contour list at the 10 largest regardless of size, which
#: silently dropped any tables past the first 10 on a page (#319). A
#: relative-area threshold scales correctly across page sizes and keeps
#: typed-character noise (a single glyph is well under 0.05% of A4) out
#: of the candidate list.
_MIN_TABLE_AREA_FRACTION = 0.0005


def undo_rotation(pdf_image, rotation):
    """Undo rotation of an image extracted from a PDF.

    Parameters
    ----------
    pdf_image: numpy.ndarray representing the image.
    rotation: str
       Either "" (no rotation), "clockwise", or "anticlockwise".  The
       **inverse** of this rotation will be applied to the image.

    Returns
    -------
    img: numpy.ndarray representing the rotated image.
    """
    if rotation == "clockwise":
        return cv2.rotate(pdf_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "anticlockwise":
        return cv2.rotate(pdf_image, cv2.ROTATE_90_CLOCKWISE)
    return pdf_image


def adaptive_threshold(
    imagename, process_background=False, blocksize=15, c=-2, rotation=""
):
    """Thresholds an image using OpenCV's adaptiveThreshold.

    Parameters
    ----------
    imagename : string
        Path to image file.
    process_background : bool, optional (default: False)
        Whether or not to process lines that are in background.
    blocksize : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    c : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    rotation: str, optional (default: "")
        Either "" (no rotation), "clockwise", or "anticlockwise".  The
        **inverse** of this rotation will be applied to the image.

    Returns
    -------
    img : object
        numpy.ndarray representing the original image.
    threshold : object
        numpy.ndarray representing the thresholded image.
    """
    img = cv2.imread(imagename)
    img = undo_rotation(img, rotation)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not process_background:
        gray = np.invert(gray)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c
    )
    return img, threshold


def find_lines(
    threshold,
    regions=None,
    direction="horizontal",
    line_scale=40,
    iterations=0,
    erode_iterations=0,
):
    """
    Finds horizontal and vertical lines by applying morphological transformations on an image.

    Parameters
    ----------
    threshold : object
        numpy.ndarray representing the thresholded image.
    regions : list, optional (default: None)
        List of page regions that may contain tables of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in image coordinate space.
    direction : string, optional (default: 'horizontal')
        Specifies whether to find vertical or horizontal lines.
    line_scale : int, optional (default: 40)
        Factor by which the page dimensions will be divided to get
        smallest length of lines that should be detected.
    iterations : int, optional (default: 0)
        Number of dilation passes applied to close small gaps in the
        line mask. Useful for tables whose ruled lines don't quite
        meet at corners.
    erode_iterations : int, optional (default: 0)
        Number of erosion passes applied **after** dilation. Set equal
        to ``iterations`` to perform morphological *closing* (dilate
        then erode of equal count): gaps are closed without enlarging
        the line mask overall. Requested in #363 — previously the
        erode step was missing, so ``iterations>=1`` widened every
        line and added phantom top/bottom lines around tables.

    Returns
    -------
    dmask : object
        numpy.ndarray representing pixels where vertical/horizontal
        lines lie.
    lines : list
        List of tuples representing vertical/horizontal lines with
        coordinates relative to a left-top origin in
        image coordinate space.
    """
    if direction not in ["vertical", "horizontal"]:
        raise ValueError("Specify direction as either 'vertical' or 'horizontal'")

    el, size = create_structuring_element(threshold, direction, line_scale)
    threshold = apply_region_mask(threshold, regions)

    processed_threshold = process_image(threshold, el, iterations, erode_iterations)
    contours, _ = cv2.findContours(
        processed_threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = extract_lines_from_contours(contours, direction)

    return processed_threshold, lines


def create_structuring_element(threshold, direction, line_scale):
    """
    Create a structuring element based on the specified direction.

    Parameters
    ----------
    threshold : object
        numpy.ndarray representing the thresholded image.
    direction : string
        Direction to create the structuring element.
    line_scale : int
        Factor for scaling the size of the structuring element.

    Returns
    -------
    tuple
        The structuring element and its size.
    """
    if direction == "vertical":
        size = threshold.shape[0] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:  # direction == "horizontal"
        size = threshold.shape[1] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    return el, size


def apply_region_mask(threshold, regions):
    """
    Apply a mask to the threshold image based on specified regions.

    Parameters
    ----------
    threshold : object
        numpy.ndarray representing the thresholded image.
    regions : list
        List of regions to apply the mask.

    Returns
    -------
    numpy.ndarray
        The masked threshold image.
    """
    if regions is not None:
        region_mask = np.zeros(threshold.shape, dtype=np.uint8)
        for region in regions:
            x, y, w, h = region
            region_mask[y : y + h, x : x + w] = 1
        threshold = np.multiply(threshold, region_mask)

    return threshold


def process_image(threshold, el, iterations, erode_iterations=0):
    """
    Apply morphological operations to the threshold image.

    Parameters
    ----------
    threshold : object
        numpy.ndarray representing the thresholded image.
    el : object
        Structuring element for morphological operations.
    iterations : int
        Number of dilation passes applied to close small gaps in the
        line mask.
    erode_iterations : int, optional (default: 0)
        Number of erosion passes applied *after* the dilation. When
        equal to ``iterations`` this is a morphological closing —
        gaps in the lines are bridged without thickening the mask
        overall. See #363.

    Returns
    -------
    numpy.ndarray
        The processed threshold image.
    """
    threshold = cv2.erode(threshold, el)
    threshold = cv2.dilate(threshold, el)
    dmask = cv2.dilate(threshold, el, iterations=iterations)
    if erode_iterations:
        dmask = cv2.erode(dmask, el, iterations=erode_iterations)

    return dmask


def extract_lines_from_contours(contours, direction):
    """
    Extract lines from contours based on the specified direction.

    Parameters
    ----------
    contours : list
        List of contours found in the image.
    direction : string
        Specifies whether to extract vertical or horizontal lines.

    Returns
    -------
    list
        List of tuples representing the coordinates of the lines.
    """
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x1, x2 = x, x + w
        y1, y2 = y, y + h
        if direction == "vertical":
            lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
        elif direction == "horizontal":
            lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))

    return lines


def find_contours(vertical, horizontal):
    """Find table boundaries using OpenCV's findContours.

    Parameters
    ----------
    vertical : object
        numpy.ndarray representing pixels where vertical lines lie.
    horizontal : object
        numpy.ndarray representing pixels where horizontal lines lie.

    Returns
    -------
    cont : list
        List of tuples representing table boundaries. Each tuple is of
        the form (x, y, w, h) where (x, y) -> left-top, w -> width and
        h -> height in image coordinate space.

    """
    mask = vertical + horizontal

    contours, __ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort largest-first so callers iterating in detected order pick up the
    # primary table first; filter out anything smaller than the configured
    # fraction of the page area. Replaces the previous arbitrary 10-contour
    # cap that silently dropped tables past index 9 (#319).
    page_area = float(mask.shape[0] * mask.shape[1]) or 1.0
    min_area = page_area * _MIN_TABLE_AREA_FRACTION
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    cont = []
    for c in contours:
        c_poly = cv2.approxPolyDP(c, 3, True)
        x, y, w, h = cv2.boundingRect(c_poly)
        cont.append((x, y, w, h))
    return cont


def find_joints(contours, vertical, horizontal):
    """Find joints/intersections present inside each table boundary.

    Parameters
    ----------
    contours : list
        List of tuples representing table boundaries. Each tuple is of
        the form (x, y, w, h) where (x, y) -> left-top, w -> width and
        h -> height in image coordinate space.
    vertical : object
        numpy.ndarray representing pixels where vertical lines lie.
    horizontal : object
        numpy.ndarray representing pixels where horizontal lines lie.

    Returns
    -------
    tables : dict
        Dict with table boundaries as keys and list of intersections
        in that boundary as their value.
        Keys are of the form (x1, y1, x2, y2) where (x1, y1) -> lb
        and (x2, y2) -> rt in image coordinate space.

    """
    joints = np.multiply(vertical, horizontal)
    tables = {}
    for c in contours:
        x, y, w, h = c
        roi = joints[y : y + h, x : x + w]
        jc, __ = cv2.findContours(
            roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(jc) <= 4:  # remove contours with less than 4 joints
            continue
        joint_coords = []
        for j in jc:
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = x + (2 * jx + jw) // 2, y + (2 * jy + jh) // 2
            joint_coords.append((c1, c2))
        tables[(x, y + h, x + w, y)] = joint_coords

    return tables


# --- Stage 1 of #763: vector-line engine for Lattice -------------------------
#
# These helpers read ruled lines directly from playa's layout tree (LTLine /
# LTRect) instead of rasterising the page and using OpenCV findContours. They
# are not wired into the Lattice parser yet — that's Stage 2 of #763, gated
# behind a new ``engine='vector'`` kwarg. Stage 1 lands the helpers + tests
# so the function shape can be reviewed in isolation.

#: Angle tolerance (in PDF units) within which a line is considered
#: orthogonal. PDFs draw nominally-axis-aligned ruled lines with the exact
#: equal coordinates we want, but a tiny float epsilon — be generous, we
#: don't want to drop a 0.0001-unit-off line just because of rounding.
_LINE_ORTHOGONAL_TOL = 0.5

#: A "filled" rectangle whose narrower side is below this width is treated
#: as a stroked line (some PDF generators draw ruled lines as 0.5pt-wide
#: filled rectangles rather than as stroked LTLines).
_LINE_AS_THIN_RECT_TOL = 1.5


def _walk_layout_objects(container):
    """Recursively yield every object in an LTContainer subtree."""
    for obj in container:
        yield obj
        if isinstance(obj, LTContainer):
            yield from _walk_layout_objects(obj)


def _ruled_lines_from_layout(layout):
    """Collect ruled line segments from a layout tree (PDF coord space).

    Returns a list of ``(x0, y0, x1, y1)`` tuples. Both stroked LTLines
    and the four edges of stroked LTRects are emitted. Filled-but-not-
    stroked LTRects with one narrow dimension are also treated as lines
    (covers PDFs that draw rules as thin filled rectangles).
    """
    out: list[tuple[float, float, float, float]] = []
    for obj in _walk_layout_objects(layout):
        if isinstance(obj, LTLine) and getattr(obj, "stroke", True):
            out.append((obj.x0, obj.y0, obj.x1, obj.y1))
        elif isinstance(obj, LTRect):
            stroked = getattr(obj, "stroke", False)
            filled = getattr(obj, "fill", False)
            if stroked:
                # Four edges of the rectangle as separate line segments.
                out.append((obj.x0, obj.y0, obj.x1, obj.y0))  # bottom
                out.append((obj.x0, obj.y1, obj.x1, obj.y1))  # top
                out.append((obj.x0, obj.y0, obj.x0, obj.y1))  # left
                out.append((obj.x1, obj.y0, obj.x1, obj.y1))  # right
            elif filled:
                # Thin filled rect → treat the long axis as the line.
                width = obj.x1 - obj.x0
                height = obj.y1 - obj.y0
                if (
                    min(width, height) <= _LINE_AS_THIN_RECT_TOL
                    and max(width, height) > _LINE_AS_THIN_RECT_TOL
                ):
                    if width >= height:
                        mid_y = (obj.y0 + obj.y1) / 2
                        out.append((obj.x0, mid_y, obj.x1, mid_y))
                    else:
                        mid_x = (obj.x0 + obj.x1) / 2
                        out.append((mid_x, obj.y0, mid_x, obj.y1))
    return out


def find_lines_from_layout(layout, direction="horizontal"):
    """Return ruled lines from a playa layout tree (Stage 1 of #763).

    Drop-in companion to :func:`find_lines` for ``flavor='lattice'``'s
    vector engine. Walks ``layout`` (recursively), classifies each
    LTLine / stroked LTRect / thin filled LTRect into horizontal or
    vertical based on its dominant axis, and returns the subset matching
    ``direction``.

    Parameters
    ----------
    layout : object
        An ``LTPage`` / ``LTContainer`` from ``playa`` — typically what
        the lattice parser already stores on ``self.layout``.
    direction : str, optional (default: 'horizontal')
        ``'horizontal'`` returns lines whose y-delta is below the
        orthogonality tolerance; ``'vertical'`` returns those whose
        x-delta is below it.

    Returns
    -------
    lines : list[tuple[float, float, float, float]]
        Each tuple is ``(x0, y0, x1, y1)`` in PDF coordinate space —
        same shape as :func:`find_lines`'s second return value, but in
        PDF coords (not image coords). The Lattice integration in
        Stage 2 will apply the existing ``scale_pdf`` to convert.

    Notes
    -----
    The output drops the morphological mask :func:`find_lines` produces
    as its first return; the vector engine doesn't need it because it
    bypasses :func:`find_contours` (which is the only consumer of the
    mask) and computes table bboxes from line intersections directly.
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    raw = _ruled_lines_from_layout(layout)
    result: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in raw:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if direction == "horizontal" and dy <= _LINE_ORTHOGONAL_TOL and dx > 0:
            result.append((min(x0, x1), y0, max(x0, x1), y0))
        elif direction == "vertical" and dx <= _LINE_ORTHOGONAL_TOL and dy > 0:
            result.append((x0, min(y0, y1), x0, max(y0, y1)))
    return result
