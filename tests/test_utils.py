"""Test to check intersection logic when no intersection area returned."""

import os

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

from camelot.utils import bbox_intersection_area


def get_text_from_pdf(filename):
    """Method to extract text object from pdf."""
    # https://stackoverflow.com/questions/22898145/how-to-extract-text-and-text-coordinates-from-a-pdf-file
    # https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html
    document = open(filename, "rb")
    # Create resource manager
    rsrcmgr = PDFResourceManager()
    # Set parameters for analysis.
    laparams = LAParams()
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(document):
        interpreter.process_page(page)
        # receive the LTPage object for the page.
        layout = device.get_result()
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
