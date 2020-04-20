# -*- coding: utf-8 -*-

import os

import pytest

import matplotlib

import camelot

# The version of Matplotlib has an impact on some of the tests.  Unfortunately,
# we can't enforce usage of a recent version of MatplotLib without dropping
# support for Python 3.6.
# To check the version of matplotlib installed:
#   pip freeze | grep matplotlib
# To force upgrade:
#   pip install --upgrade --force-reinstall matplotlib
# To force usage of a Python 3.6 compatible version:
#   pip install "matplotlib==2.2.5"
# This condition can be removed in favor of a version requirement bump for
# matplotlib once support for Python 3.5 is dropped.

LEGACY_MATPLOTLIB = matplotlib.__version__ < "3.2.1"

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_text_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='text')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='grid')

@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind='grid')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_hybrid_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, flavor="hybrid")
    return camelot.plot(tables[0], kind='grid')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='contour')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return camelot.plot(tables[0], kind='contour')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_hybrid_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='hybrid')
    return camelot.plot(tables[0], kind='contour')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='line')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind='joint')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return camelot.plot(tables[0], kind='textedge')


@pytest.mark.skipif(LEGACY_MATPLOTLIB,
                    reason="depends on a recent version of MatPlotLib")
@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True)
def test_hybrid_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, debug=True, flavor='hybrid')
    return camelot.plot(tables[0], kind='textedge')
