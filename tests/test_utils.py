"""Test to check intersection logic when no intersection area returned."""

import os

import paves.miner as pm
from paves.miner import LAParams
from paves.miner import LTTextBoxHorizontal

from camelot.utils import bbox_intersection_area


def get_text_from_pdf(filename):
    """Method to extract text object from pdf."""
    for layout in pm.extract(filename, LAParams()):
        for element in layout:
            if isinstance(element, LTTextBoxHorizontal):
                return element


def test_bbox_intersection_text(testdir):
    """
    Test to check area of intersection between both boxes when no intersection area returned.
    """
    filename1 = os.path.join(testdir, "foo.pdf")
    pdftextelement1 = get_text_from_pdf(filename1)
    filename2 = os.path.join(testdir, "tabula/12s0324.pdf")
    pdftextelement2 = get_text_from_pdf(filename2)

    assert bbox_intersection_area(pdftextelement1, pdftextelement2) == 0.0
