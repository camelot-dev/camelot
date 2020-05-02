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
#   pip install "matplotlib==3.0.3"
# This condition can be removed in favor of a version requirement bump for
# matplotlib once support for Python 3.5 is dropped.

LEGACY_MATPLOTLIB = matplotlib.__version__ < "3.2.1"

# Bump the default plot tolerance from 2 to account for cross-platform testing
# via Travis, and resulting minor font changes.
TOLERANCE = 4

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


def unit_test_stable_plot(table, kind):
    if not LEGACY_MATPLOTLIB:
        # See https://matplotlib.org/3.2.1/users/whats_new.html#kerning-adjustments-now-use-correct-values  # noqa
        matplotlib.rcParams["text.kerning_factor"] = 6
    return camelot.plot(table, kind=kind)


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_text_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return unit_test_stable_plot(tables[0], 'text')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return unit_test_stable_plot(tables[0], 'grid')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_stream_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return unit_test_stable_plot(tables[0], 'grid')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_network_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, flavor="network")
    return unit_test_stable_plot(tables[0], 'grid')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_lattice_contour_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return unit_test_stable_plot(tables[0], 'contour')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_stream_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return unit_test_stable_plot(tables[0], 'contour')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_network_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='network')
    return unit_test_stable_plot(tables[0], 'contour')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_line_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return unit_test_stable_plot(tables[0], 'line')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_joint_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return unit_test_stable_plot(tables[0], 'joint')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_stream_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor='stream')
    return unit_test_stable_plot(tables[0], 'textedge')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_network_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, debug=True, flavor='network')
    return unit_test_stable_plot(tables[0], 'textedge')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_network_table_regions_textedge_plot():
    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, debug=True, flavor="network",
        table_regions=["320,505,573,330"]
    )
    return unit_test_stable_plot(tables[0], 'textedge')


@pytest.mark.mpl_image_compare(
    baseline_dir="files/baseline_plots", remove_text=True, tolerance=TOLERANCE)
def test_network_table_areas_text_plot():
    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, debug=True, flavor="network",
        table_areas=["320,500,573,335"]
    )
    return unit_test_stable_plot(tables[0], 'text')
