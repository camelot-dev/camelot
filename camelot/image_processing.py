"""Functions useful for detecting graphical elements from the image to using OpenCV to reconstruct / detect tables."""

import cv2
import numpy as np


def adaptive_threshold(imagename, process_background=False, blocksize=15, c=-2):
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

    Returns
    -------
    img : object
        numpy.ndarray representing the original image.
    threshold : object
        numpy.ndarray representing the thresholded image.
    """
    img = cv2.imread(imagename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not process_background:
        gray = np.invert(gray)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c
    )
    return img, threshold


def find_lines(
    threshold, regions=None, direction="horizontal", line_scale=40, iterations=0
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
        Number of times for erosion/dilation is applied.

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

    processed_threshold = process_image(threshold, el, iterations)
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


def process_image(threshold, el, iterations):
    """
    Apply morphological operations to the threshold image.

    Parameters
    ----------
    threshold : object
        numpy.ndarray representing the thresholded image.
    el : object
        Structuring element for morphological operations.
    iterations : int
        Number of iterations for dilation.

    Returns
    -------
    numpy.ndarray
        The processed threshold image.
    """
    threshold = cv2.erode(threshold, el)
    threshold = cv2.dilate(threshold, el)
    dmask = cv2.dilate(threshold, el, iterations=iterations)

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
    # sort in reverse based on contour area and use first 10 contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

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
