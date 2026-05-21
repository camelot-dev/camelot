"""read_pdf accepting bytes / BinaryIO / file-like (#170, #245, #270)."""

import io
import os

import camelot


def test_read_pdf_from_bytes(testdir):
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as f:
        data = f.read()
    tables = camelot.read_pdf(data, flavor="lattice")
    assert len(tables) == 1


def test_read_pdf_from_bytesio(testdir):
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as f:
        bio = io.BytesIO(f.read())
    tables = camelot.read_pdf(bio, flavor="lattice")
    assert len(tables) == 1


def test_read_pdf_from_open_file_handle(testdir):
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as fh:
        tables = camelot.read_pdf(fh, flavor="lattice")
    assert len(tables) == 1


def test_read_pdf_bytesio_cursor_restored(testdir):
    """We restore the stream's read position so the caller can keep using it."""
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as f:
        bio = io.BytesIO(f.read())
    bio.seek(10)
    _ = camelot.read_pdf(bio, flavor="lattice")
    assert bio.tell() == 10


def test_read_pdf_from_bytes_works_with_stream_flavor(testdir):
    src = os.path.join(testdir, "tabula/us-007.pdf")
    with open(src, "rb") as f:
        data = f.read()
    # stream doesn't need image conversion — exercises the playa-only path.
    tables = camelot.read_pdf(data, flavor="stream", table_areas=["320,500,573,335"])
    assert len(tables) >= 1


def test_read_pdf_bytearray_accepted(testdir):
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as f:
        data = bytearray(f.read())
    tables = camelot.read_pdf(data, flavor="lattice")
    assert len(tables) == 1


def test_handler_close_removes_bytes_tempfile(testdir):
    """The spilled tempfile is reaped when the context manager exits."""
    src = os.path.join(testdir, "foo.pdf")
    with open(src, "rb") as f:
        data = f.read()
    from camelot.handlers import PDFHandler

    with PDFHandler(data) as p:
        tempname = p.filepath
        assert os.path.exists(tempname)
    assert not os.path.exists(tempname)
