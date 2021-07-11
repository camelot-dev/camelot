# -*- coding: utf-8 -*-

import os
import sys

import pytest

import camelot

testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")

skip_on_windows = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Ghostscript not installed in Windows test environment",
)


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_text_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="text")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="textedge")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot_poppler():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="contour")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot_ghostscript():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="contour")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="contour")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot_poppler():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="line")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot_ghostscript():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="line")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot_poppler():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="joint")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot_ghostscript():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="joint")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot_poppler():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="grid")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot_ghostscript():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="grid")
