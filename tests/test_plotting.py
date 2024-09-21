import os

import pytest

import camelot
from tests.conftest import skip_on_windows
from tests.conftest import skip_pdftopng


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_text_plot(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="text")

@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_textedge_plot(testdir):
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="textedge")


@skip_pdftopng
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot_poppler(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="contour")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot_ghostscript(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="contour")

@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_contour_plot(testdir):
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="contour")


@skip_pdftopng
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot_poppler(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="line")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot_ghostscript(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="line")


@skip_pdftopng
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot_poppler(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="joint")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot_ghostscript(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="joint")


@skip_pdftopng
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot_poppler(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="poppler")
    return camelot.plot(tables[0], kind="grid")


@skip_on_windows
@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot_ghostscript(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, backend="ghostscript")
    return camelot.plot(tables[0], kind="grid")
