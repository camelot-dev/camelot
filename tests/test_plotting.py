# -*- coding: utf-8 -*-

import os

import pytest

import pdfminer

import camelot

# The version of PDFMiner has an impact on some of the tests.  Unfortunately,
# we can't enforce usage of a recent version of PDFMiner without dropping
# support for Python 2.
# To check the version of pdfminer.six installed:
#   pip freeze | grep pdfminer.six
# To force upgrade:
#   pip install --upgrade --force-reinstall pdfminer.six
# To force usage of a Python 2 compatible version:
#   pip install "pdfminer.six==20191110"
# This condition can be removed in favor of a version requirement bump for
# pdfminer.six once support for Python 2 is dropped.

LEGACY_PDF_MINER = pdfminer.__version__ < "20200402"

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


@pytest.mark.skipif(LEGACY_PDF_MINER,
                    reason="depends on a recent version of PDFMiner")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_text_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='text')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='grid')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='contour')


@pytest.mark.skipif(LEGACY_PDF_MINER,
                    reason="depends on a recent version of PDFMiner")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return camelot.plot(tables[0], kind='contour')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='line')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='joint')


@pytest.mark.skipif(LEGACY_PDF_MINER,
                    reason="depends on a recent version of PDFMiner")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return camelot.plot(tables[0], kind='textedge')
