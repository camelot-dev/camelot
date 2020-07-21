# -*- coding: utf-8 -*-

import os
import sys
import logging

from PyPDF2 import PdfFileReader, PdfFileWriter

from .core import TableList
from .parsers import Stream, Lattice, Network, Hybrid
from .utils import (
    build_file_path_in_temp_dir,
    get_page_layout,
    get_text_objects,
    get_rotation,
    is_url,
    download_url,
)

logger = logging.getLogger("camelot")

PARSERS = {
    "lattice": Lattice,
    "stream": Stream,
    "network": Network,
    "hybrid": Hybrid,
}


class PDFHandler():
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
    debug : bool, optional (default: False)
        Whether the parser should store debug information during parsing.

    """

    def __init__(self, filepath, pages="1", password=None, debug=False):
        self.debug = debug
        if is_url(filepath):
            filepath = download_url(filepath)
        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""
        else:
            self.password = password
            if sys.version_info[0] < 3:
                self.password = self.password.encode("ascii")
        self.pages = self._get_pages(self.filepath, pages)

    def _get_pages(self, filepath, pages):
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
            infile = PdfFileReader(open(filepath, "rb"), strict=False)
            if infile.isEncrypted:
                infile.decrypt(self.password)
            if pages == "all":
                page_numbers.append({"start": 1, "end": infile.getNumPages()})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = infile.getNumPages()
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})
        P = []
        for p in page_numbers:
            P.extend(range(p["start"], p["end"] + 1))
        return sorted(set(P))

    def _read_pdf_page(self, page=1, layout_kwargs=None):
        """Saves specified page from PDF into a temporary directory. Removes
        password protection and normalizes rotation.

        Parameters
        ----------
        page : int
            Page number.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.  # noqa


        Returns
        -------
        layout : object

        dimensions : tuple
            The dimensions of the pdf page

        filepath : str
            The path of the single page PDF - either the original, or a
            normalized version.

        """
        layout_kwargs = layout_kwargs or {}
        with open(self.filepath, "rb") as fileobj:
            # Normalize the pdf file, but skip if it's not encrypted or has
            # only one page.
            infile = PdfFileReader(fileobj, strict=False)
            if infile.isEncrypted:
                infile.decrypt(self.password)
            fpath = build_file_path_in_temp_dir(f"page-{page}.pdf")
            froot, fext = os.path.splitext(fpath)
            p = infile.getPage(page - 1)
            outfile = PdfFileWriter()
            outfile.addPage(p)
            with open(fpath, "wb") as f:
                outfile.write(f)
            layout, dimensions = get_page_layout(
                fpath, **layout_kwargs)
            # fix rotated PDF
            chars = get_text_objects(layout, ltype="char")
            horizontal_text = get_text_objects(layout, ltype="horizontal_text")
            vertical_text = get_text_objects(layout, ltype="vertical_text")
            rotation = get_rotation(chars, horizontal_text, vertical_text)
            if rotation != "":
                fpath_new = "".join(
                    [froot.replace("page", "p"), "_rotated", fext])
                os.rename(fpath, fpath_new)
                infile = PdfFileReader(open(fpath_new, "rb"), strict=False)
                if infile.isEncrypted:
                    infile.decrypt(self.password)
                outfile = PdfFileWriter()
                p = infile.getPage(0)
                if rotation == "anticlockwise":
                    p.rotateClockwise(90)
                elif rotation == "clockwise":
                    p.rotateCounterClockwise(90)
                outfile.addPage(p)
                with open(fpath, "wb") as f:
                    outfile.write(f)
                layout, dimensions = get_page_layout(
                    fpath, **layout_kwargs)
        return layout, dimensions, fpath

    def parse(
        self, flavor="lattice", suppress_stdout=False,
        layout_kwargs=None, **kwargs
    ):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice', 'stream', 'network',
            or 'hybrid').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs. # noqa
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        layout_kwargs = layout_kwargs or {}
        tables = []

        parser_obj = PARSERS[flavor]
        parser = parser_obj(debug=self.debug, **kwargs)

        # Read the layouts/dimensions of each of the pages we need to
        # parse. This might require creating a temporary .pdf.
        for page_idx in self.pages:
            layout, dimensions, source_file = self._read_pdf_page(
                page_idx,
                layout_kwargs=layout_kwargs
            )
            parser.prepare_page_parse(source_file, layout, dimensions,
                                      page_idx, layout_kwargs)
            if not suppress_stdout:
                rootname = os.path.basename(parser.rootname)
                logger.info(f"Processing {rootname}")
            t = parser.extract_tables()
            tables.extend(t)
        return TableList(sorted(tables))
