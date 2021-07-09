import os
import warnings
from camelot.utils import is_url

import pytest

import camelot
from tests.conftest import skip_on_windows


def test_unknown_flavor(foo_pdf):
    message = "Unknown flavor specified." " Use either 'lattice' or 'stream'"
    with pytest.raises(NotImplementedError, match=message):
        camelot.read_pdf(foo_pdf, flavor="chocolate")


def test_input_kwargs(foo_pdf):
    message = "columns cannot be used with flavor='lattice'"
    with pytest.raises(ValueError, match=message):
        camelot.read_pdf(foo_pdf, columns=["10,20,30,40"])


def test_unsupported_format(testdir):
    message = "File format not supported"
    filename = os.path.join(testdir, "foo.csv")
    with pytest.raises(NotImplementedError, match=message):
        camelot.read_pdf(filename)


@skip_on_windows
def test_no_tables_found_logs_suppressed(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        try:
            camelot.read_pdf(filename, suppress_stdout=True)
        except Warning as e:
            warning_text = str(e)
            pytest.fail(f"Unexpected warning: {warning_text}")


def test_no_tables_found_warnings_suppressed(testdir):
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        try:
            camelot.read_pdf(filename, suppress_stdout=True)
        except Warning as e:
            warning_text = str(e)
            pytest.fail(f"Unexpected warning: {warning_text}")


def test_no_password(testdir):
    filename = os.path.join(testdir, "health_protected.pdf")
    message = "File has not been decrypted"
    with pytest.raises(Exception, match=message):
        camelot.read_pdf(filename)


def test_bad_password(testdir):
    filename = os.path.join(testdir, "health_protected.pdf")
    message = "File has not been decrypted"
    with pytest.raises(Exception, match=message):
        camelot.read_pdf(filename, password="wrongpass")


def test_stream_equal_length(foo_pdf):
    message = "Length of table_areas and columns" " should be equal"
    with pytest.raises(ValueError, match=message):
        camelot.read_pdf(
            foo_pdf,
            flavor="stream",
            table_areas=["10,20,30,40"],
            columns=["10,20,30,40", "10,20,30,40"],
        )


def test_image_warning(testdir):
    filename = os.path.join(testdir, "image.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        with pytest.raises(UserWarning) as e:
            camelot.read_pdf(filename)
            assert (
                str(e.value)
                == "page-1 is image-based, camelot only works on text-based pages."
            )


def test_stream_no_tables_on_page(testdir):
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            camelot.read_pdf(filename, flavor="stream")
        assert str(e.value) == "No tables found on page-1"


def test_stream_no_tables_in_area(testdir):
    filename = os.path.join(testdir, "only_page_number.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="stream")
        assert str(e.value) == "No tables found in table area 1"


def test_lattice_no_tables_on_page(testdir):
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="lattice")
        assert str(e.value) == "No tables found on page-1"


def test_lattice_unknown_backend(foo_pdf):
    message = "Unknown backend 'mupdf' specified. Please use either 'poppler' or 'ghostscript'."
    with pytest.raises(NotImplementedError, match=message):
        tables = camelot.read_pdf(foo_pdf, backend="mupdf")


def test_lattice_no_convert_method(foo_pdf):
    class ConversionBackend:
        pass

    message = "must implement a 'convert' method"
    with pytest.raises(NotImplementedError, match=message):
        camelot.read_pdf(foo_pdf, backend=ConversionBackend())


def test_lattice_ghostscript_deprecation_warning(foo_pdf):
    ghostscript_deprecation_warning = (
        "'ghostscript' will be replaced by 'poppler' as the default image conversion"
        " backend in v0.12.0. You can try out 'poppler' with backend='poppler'."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning) as e:
            camelot.read_pdf(foo_pdf)
            assert str(e.value) == ghostscript_deprecation_warning


def test_invalid_url():
    url = 'fttp://google.com/pdf'
    message = "File format not supported"
    with pytest.raises(Exception, match=message):
        url = camelot.read_pdf(url)
    assert is_url(url) == False
