# -*- coding: utf-8 -*-

import os
import warnings

import pytest

import camelot


testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")
filename = os.path.join(testdir, "foo.pdf")


def test_unknown_flavor():
    message = "Unknown flavor specified." " Use either 'lattice' or 'stream'"
    with pytest.raises(NotImplementedError, match=message):
        tables = camelot.read_pdf(filename, flavor="chocolate")


def test_input_kwargs():
    message = "columns cannot be used with flavor='lattice'"
    with pytest.raises(ValueError, match=message):
        tables = camelot.read_pdf(filename, columns=["10,20,30,40"])


def test_unsupported_format():
    message = "File format not supported"
    filename = os.path.join(testdir, "foo.csv")
    with pytest.raises(NotImplementedError, match=message):
        tables = camelot.read_pdf(filename)


def test_stream_equal_length():
    message = "Length of table_areas and columns" " should be equal"
    with pytest.raises(ValueError, match=message):
        tables = camelot.read_pdf(
            filename,
            flavor="stream",
            table_areas=["10,20,30,40"],
            columns=["10,20,30,40", "10,20,30,40"],
        )


def test_image_warning():
    filename = os.path.join(testdir, "image.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename)
            assert (
                str(e.value)
                == "page-1 is image-based, camelot only works on text-based pages."
            )


def test_lattice_no_tables_on_page():
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="lattice")
        assert str(e.value) == "No tables found on page-1"


def test_stream_no_tables_on_page():
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="stream")
        assert str(e.value) == "No tables found on page-1"


def test_stream_no_tables_in_area():
    filename = os.path.join(testdir, "only_page_number.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="stream")
        assert str(e.value) == "No tables found in table area 1"


def test_no_tables_found_logs_suppressed():
    filename = os.path.join(testdir, "foo.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        try:
            tables = camelot.read_pdf(filename, suppress_stdout=True)
        except Warning as e:
            warning_text = str(e)
            pytest.fail(f"Unexpected warning: {warning_text}")


def test_no_tables_found_warnings_suppressed():
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        try:
            tables = camelot.read_pdf(filename, suppress_stdout=True)
        except Warning as e:
            warning_text = str(e)
            pytest.fail(f"Unexpected warning: {warning_text}")


def test_no_password():
    filename = os.path.join(testdir, "health_protected.pdf")
    message = "file has not been decrypted"
    with pytest.raises(Exception, match=message):
        tables = camelot.read_pdf(filename)


def test_bad_password():
    filename = os.path.join(testdir, "health_protected.pdf")
    message = "file has not been decrypted"
    with pytest.raises(Exception, match=message):
        tables = camelot.read_pdf(filename, password="wrongpass")
