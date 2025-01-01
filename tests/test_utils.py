"""Test to check intersection logic when no intersection area returned."""

import os

import pytest
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

from camelot.utils import bbox_intersection_area, compute_whitespace, merge_close_lines


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


@pytest.mark.parametrize(
    "table_data, expected",
    [
        ([], 0.0),
        ([[]], 0.0),
        ([[""]], 100.0),
        ([["a", "b", "c"], ["d", "e", "f"]], 0.0),
        ([["", "", ""], ["", ""]], 100.0),
        ([["a", "", "c"], ["", "e", ""]], 50.0),
        ([["", "", ""]], 100.0),
        ([["a", "b", "c"]], 0.0),
        ([["a"]], 0.0),
        ([[""]], 100.0),
        (["not a list", ["a", "b"], ["", ""]], 50.0),
        ([["a", "", "42"], ["", "None", ""]], 50.0),
    ]
)
def test_compute_whitespace(table_data: list[list[str]], expected: float) -> None:
    result: float = compute_whitespace(table_data)
    assert result == pytest.approx(expected)


def test_merge_empty_list() -> None:
    """Test merging an empty list."""
    assert merge_close_lines([]) == []

def test_merge_single_element() -> None:
    """Test merging a single-element list."""
    assert merge_close_lines([5.0]) == pytest.approx([5.0])

def test_merge_no_close_elements() -> None:
    """Test a list with no close elements."""
    assert merge_close_lines([1.0, 10.0, 20.0], line_tol=2) == pytest.approx([1.0, 10.0, 20.0])

def test_merge_all_close_elements() -> None:
    """Test a list where all elements are close."""
    assert merge_close_lines([1.0, 2.0, 3.0], line_tol=2) == pytest.approx([2.25])

def test_merge_some_close_elements() -> None:
    """Test a list where some elements are close."""
    assert merge_close_lines([1.0, 2.0, 5.0, 7.0], line_tol=2) == pytest.approx([1.5, 6.0])

def test_merge_edge_case_tolerance_boundary() -> None:
    """Test merging elements exactly on the tolerance boundary."""
    assert merge_close_lines([1.0, 3.0, 5.0], line_tol=2) == pytest.approx([2.0, 5.0])

def test_merge_negative_numbers():
    """Test merging with negative numbers."""
    assert merge_close_lines([-10.0, -8.0, 1.0, 3.0], line_tol=2) == pytest.approx([-9.0, 2.0])
