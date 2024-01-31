import os
import shutil
import sys
import tempfile
import weakref
from pathlib import Path
from typing import Union

from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf._utils import StrByteType

from .core import TableList
from .parsers import Lattice
from .parsers import Stream
from .utils import download_url
from .utils import get_page_layout
from .utils import get_rotation
from .utils import get_text_objects
from .utils import is_url


class PDFHandler:
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.

    """

    def __init__(self, filepath: Union[StrByteType, Path], pages="1", password=None):
        self.tempdir = tempfile.mkdtemp()
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.tempdir)
        if is_url(filepath):
            filepath = download_url(filepath)
        self.filepath: Union[StrByteType, Path] = filepath

        if isinstance(filepath, str) and not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""  # noqa: S105
        else:
            self.password = password
            if sys.version_info[0] < 3:
                self.password = self.password.encode("ascii")
        self.pages = self._get_pages(pages)

    def _get_pages(self, pages):
        """Converts pages string to list of ints.

        Parameters
        ----------
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        result : list[int]
            List of int page numbers.

        """
        page_numbers = []

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            infile = PdfReader(self.filepath, strict=False)

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

        result = []
        for p in page_numbers:
            result.extend(range(p["start"], p["end"] + 1))
        return sorted(set(result))

    def _get_temp_path(self, page, rotated=False):
        """Generate page path with temp directory.

        Parameters
        ----------
        page : int
            Page number.
        rotated: bool (default: False)
            Switch to generate temp file name.
        """
        if rotated is False:
            return os.path.join(self.tempdir, f"page-{page}.pdf")
        else:
            return os.path.join(self.tempdir, f"p-{page}_rotated.pdf")

    def _save_page(self, page):
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        page : int
            Page number.
        """
        infile = PdfReader(self.filepath, strict=False)
        if infile.is_encrypted:
            infile.decrypt(self.password)
        fpath = self._get_temp_path(page)
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
            fpath_new = self._get_temp_path(page, rotated=True)
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
        self, flavor="lattice", suppress_stdout=False, layout_kwargs=None, **kwargs
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
            A dict of `pdfminer.layout.LAParams
            <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        if layout_kwargs is None:
            layout_kwargs = {}

        tables = []
        for p in self.pages:
            self._save_page(p)
        pages = [self._get_temp_path(p) for p in self.pages]
        parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
        for p in pages:
            t = parser.extract_tables(
                p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
            )
            tables.extend(t)
        return TableList(sorted(tables))
