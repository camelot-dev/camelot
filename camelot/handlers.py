# -*- coding: utf-8 -*-

from contextlib import contextmanager
import io
import os
import sys

from pypdf import PdfReader, PdfWriter

from .core import TableList
from .parsers import Stream, Lattice
from .utils import (
    InvalidArguments,
    TemporaryDirectory,
    get_page_layout,
    get_text_objects,
    get_rotation,
    is_url,
    get_url_bytes,
)


class PDFHandler(object):
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str | pathlib.Path, optional (default: None)
        Filepath or URL of the PDF file. Required if file_bytes is not given
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    file_bytes : io.IOBase, optional (default: None)
        A file-like stream. Required if filepath is not given

    """

    def __init__(self, filepath=None, pages="1", password=None, file_bytes=None):
        if is_url(filepath):
            file_bytes = get_url_bytes(filepath)

        if not filepath and not file_bytes:
            raise InvalidArguments('Either `filepath` or `file_bytes` is required')
        if not filepath:
            # filepath must either be passed, or taken from the name attribute
            filepath = getattr(file_bytes, 'name')
            if not filepath:
                msg = ('Either pass a `filepath`, or give the '
                       '`file_bytes` argument a name attribute')
                raise InvalidArguments(msg)
        self.file_bytes = file_bytes  # ok to be None

        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""
        else:
            self.password = password
            if sys.version_info[0] < 3:
                self.password = self.password.encode("ascii")
        self.pages = self._get_pages(pages)

    @contextmanager
    def managed_file_context(self):
        """Reads from either the `filepath` or `file_bytes`
        attribute of this instance, to return a file-like object.
        Closes any open file handles on exit or error.

        Returns
        -------
        file_bytes : io.IOBase
            A readable, seekable, file-like object
        """
        if self.file_bytes:
            # if we can't seek, write to a BytesIO object that can,
            # then seek to the beginning before yielding
            if not hasattr(self.file_bytes, 'seek'):
                self.file_bytes = io.BytesIO(self.file_bytes.read())
            self.file_bytes.seek(0)
            yield self.file_bytes
        else:
            with open(self.filepath, "rb") as file_bytes:
                yield file_bytes

    def _get_pages(self, pages):
        """Converts pages string to list of ints.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        P : list
            List of int page numbers.

        """
        page_numbers = []

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            with self.managed_file_context() as f:
                infile = PdfReader(f, strict=False)

                if infile.is_encrypted:
                    infile.decrypt(self.password)

                if pages == "all":
                    page_numbers.append({"start": 1, "end": len(infile.pages)})
                else:
                    for r in pages.split(","):
                        if "-" in r:
                            a, b = r.split("-")
                            if b == "end":
                                b = len(infile.pages)
                            page_numbers.append({"start": int(a), "end": int(b)})
                        else:
                            page_numbers.append({"start": int(r), "end": int(r)})

        P = []
        for p in page_numbers:
            P.extend(range(p["start"], p["end"] + 1))
        return sorted(set(P))

    def _save_page(self, filepath, page, temp):
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        temp : str
            Tmp directory.

        """
        with self.managed_file_context() as fileobj:
            infile = PdfReader(fileobj, strict=False)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            fpath = os.path.join(temp, f"page-{page}.pdf")
            froot, fext = os.path.splitext(fpath)
            p = infile.pages[page - 1]
            outfile = PdfWriter()
            outfile.add_page(p)
            with open(fpath, "wb") as f:
                outfile.write(f)
            layout, dim = get_page_layout(fpath)
            # fix rotated PDF
            chars = get_text_objects(layout, ltype="char")
            horizontal_text = get_text_objects(layout, ltype="horizontal_text")
            vertical_text = get_text_objects(layout, ltype="vertical_text")
            rotation = get_rotation(chars, horizontal_text, vertical_text)
            if rotation != "":
                fpath_new = "".join([froot.replace("page", "p"), "_rotated", fext])
                os.rename(fpath, fpath_new)
                instream = open(fpath_new, "rb")
                infile = PdfReader(instream, strict=False)
                if infile.is_encrypted:
                    infile.decrypt(self.password)
                outfile = PdfWriter()
                p = infile.pages[0]
                if rotation == "anticlockwise":
                    p.rotate(90)
                elif rotation == "clockwise":
                    p.rotate(-90)
                outfile.add_page(p)
                with open(fpath, "wb") as f:
                    outfile.write(f)
                instream.close()

    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        tables = []
        with TemporaryDirectory() as tempdir:
            for p in self.pages:
                self._save_page(self.filepath, p, tempdir)
            pages = [os.path.join(tempdir, f"page-{p}.pdf") for p in self.pages]
            parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
            for p in pages:
                t = parser.extract_tables(
                    p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
                )
                tables.extend(t)
        return TableList(sorted(tables))
