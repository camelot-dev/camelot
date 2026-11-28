import os
import sys
import warnings
from unittest import mock

import pytest
from playa.exceptions import PDFTextExtractionNotAllowed

import camelot
from camelot.backends import pdfium_backend
from camelot.backends.image_conversion import ImageConversionError
from camelot.utils import is_url
from tests.conftest import skip_on_windows


def test_unknown_flavor(foo_pdf):
    message = (
        "Unknown flavor specified."
        " Use either 'lattice', 'stream', 'network', 'hybrid', 'ml' or 'auto'"
    )
    with pytest.raises(NotImplementedError, match=message):
        camelot.read_pdf(foo_pdf, flavor="chocolate")


def test_input_kwargs(foo_pdf):
    message = "columns cannot be used with flavor='lattice'"
    with pytest.raises(ValueError, match=message):
        camelot.read_pdf(foo_pdf, columns=["10,20,30,40"])


def test_pdf_without_pdf_extension(testdir, tmp_path):
    """Non-`.pdf` filenames containing valid PDF bytes are accepted (#646).

    NamedTemporaryFile and similar producers commonly omit the suffix; the
    file-extension check used to reject them even though the bytes parse
    fine. Symmetric with the Path-arg path, which already bypassed it.
    """
    src = os.path.join(testdir, "foo.pdf")
    dst = tmp_path / "no_extension"
    dst.write_bytes(open(src, "rb").read())
    tables = camelot.read_pdf(str(dst))
    assert len(tables) == 1


@skip_on_windows
def test_no_tables_found_logs_suppressed(testdir):
    filename = os.path.join(testdir, "foo.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        tables = camelot.read_pdf(filename, suppress_stdout=True)
    assert len(tables) == 1


def test_no_tables_found_warnings_suppressed(testdir):
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        # the test should fail if any warning is thrown
        warnings.simplefilter("error")
        tables = camelot.read_pdf(filename, suppress_stdout=True)
    assert len(tables) == 0


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
    message = "Length of table_areas and columns should be equal"
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
        assert str(e.value) == "No tables found in table area (0, 0, 792.0, 612.0)"


HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def _pdf_with_text(tmp_path, width, height, x, y, text, font=HELVETICA):
    """A one-page PDF that has `text` at (x, y), written with Helvetica."""
    content = f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ).encode(),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        font,
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        pdf += b"%010d 00000 n \n" % offset
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path = tmp_path / "text.pdf"
    path.write_bytes(bytes(pdf))
    return str(path)


@pytest.mark.parametrize(
    ("size", "x", "text"),
    [
        pytest.param(200, 50, " ", id="only-whitespace"),
        pytest.param(5, 50, "Background", id="text-outside-the-page"),
    ],
)
def test_stream_area_without_text(tmp_path, size, x, text):
    """An area without text, or with only whitespace, has no table to build."""
    filename = _pdf_with_text(tmp_path, size, size, x, 50, text)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning) as e:
            camelot.read_pdf(filename, flavor="stream")
        assert (
            str(e.value) == f"No tables found in table area (0, 0, {size}.0, {size}.0)"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(camelot.read_pdf(filename, flavor="stream")) == 0


def test_stream_text_without_width(tmp_path):
    """Text whose glyphs have no width makes a column without width."""
    widths = " ".join(["0"] * 95)
    font = (
        "<< /Type /Font /Subtype /Type1 /BaseFont /Foo "
        f"/FirstChar 32 /LastChar 126 /Widths [{widths}] >>"
    ).encode()
    filename = _pdf_with_text(tmp_path, 200, 200, 50, 50, "Lorem ipsum", font)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = camelot.read_pdf(filename, flavor="stream")
    assert len(tables) == 1
    assert "".join(tables[0].df[0][0].split()) == "Loremipsum"


def test_lattice_no_tables_on_page(testdir):
    filename = os.path.join(testdir, "empty.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        with pytest.raises(UserWarning) as e:
            tables = camelot.read_pdf(filename, flavor="lattice")
        assert str(e.value) == "No tables found on page-1"


def test_lattice_unknown_backend(foo_pdf):
    message = "Unknown backend 'mupdf' specified. Please use 'pdfium', 'poppler' or 'ghostscript'."
    with pytest.raises(NotImplementedError, match=message):
        tables = camelot.read_pdf(
            foo_pdf, flavor="lattice", backend="mupdf", use_fallback=False
        )


def test_lattice_no_convert_method(foo_pdf):
    class BackendWithoutConvert:
        """Backend that deliberately lacks the required convert() method."""

    message = "must implement a 'convert' method"
    with pytest.raises(NotImplementedError, match=message):
        camelot.read_pdf(
            foo_pdf,
            flavor="lattice",
            backend=BackendWithoutConvert(),
            use_fallback=False,
        )


def test_invalid_url():
    # `fttp://` is not a recognised URL scheme, so is_url returns False and
    # the string is treated as a file path. Since no such file exists, a
    # FileNotFoundError is raised (rather than the old extension-check
    # NotImplementedError, which was removed in #646).
    url = "fttp://google.com/pdf"
    assert is_url(url) is False
    with pytest.raises((FileNotFoundError, OSError)):
        camelot.read_pdf(url)


def test_pdfium_backend_import_error(testdir, tmp_path, monkeypatch):
    filename = os.path.join(testdir, "table_region.pdf")
    # pypdfium2 is imported when camelot loads, so patching sys.modules here
    # has no effect; simulate the failed import the backend records instead.
    monkeypatch.setattr(
        pdfium_backend,
        "PDFIUM_EXC",
        ModuleNotFoundError("No module named 'pypdfium2'"),
    )
    message = "pypdfium2 is not available: "
    with pytest.raises(OSError, match=message):
        pdfium_backend.PdfiumBackend().convert(filename, str(tmp_path / "out.png"))


def test_pdfium_backend_import_error_alternative(testdir):
    filename = os.path.join(testdir, "table_region.pdf")
    with mock.patch.dict(sys.modules, {"pypdfium2": None}):
        message = "pypdfium2 is not available: "
        tables = camelot.read_pdf(
            filename,
            flavor="lattice",
            backend="pdfium",
            use_fallback=False,
        )
    assert tables is not None


def test_ghostscript_backend_import_error(testdir):
    filename = os.path.join(testdir, "table_region.pdf")
    with mock.patch.dict(sys.modules, {"ghostscript": None}):
        message = "Ghostscript is not installed"
        with pytest.raises(ImageConversionError) as e:
            tables = camelot.read_pdf(
                filename,
                flavor="lattice",
                backend="ghostscript",
                use_fallback=False,
            )
        assert message in str(e.value)


def test_extraction_not_allowed(testdir):
    filename = os.path.join(testdir, "birdisland.pdf")
    with pytest.raises(
        PDFTextExtractionNotAllowed, match="Text extraction is not allowed"
    ):
        camelot.read_pdf(filename, flavor="stream")
