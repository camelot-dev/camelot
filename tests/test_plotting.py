# -*- coding: utf-8 -*-

import os

import pytest

import camelot


testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_text_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="text")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_grid_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="grid")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_lattice_contour_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="contour")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_stream_contour_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="contour")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_line_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="line")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_joint_plot():
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename)
    return camelot.plot(tables[0], kind="joint")


@pytest.mark.mpl_image_compare(baseline_dir="files/baseline_plots", remove_text=True)
def test_textedge_plot():
    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    return camelot.plot(tables[0], kind="textedge")
