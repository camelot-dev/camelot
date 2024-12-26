"""Functions to handle all operations on the PDF's."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from pdfminer.layout import LTChar
from pdfminer.layout import LTImage
from pdfminer.layout import LTTextLineHorizontal
from pdfminer.layout import LTTextLineVertical
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf._utils import StrByteType

from .core import TableList
from .parsers import Hybrid
from .parsers import Lattice
from .parsers import Network
from .parsers import Stream
from .utils import TemporaryDirectory
from .utils import download_url
from .utils import get_image_char_and_text_objects
from .utils import get_page_layout
from .utils import get_rotation
from .utils import is_url


PARSERS = {
    "lattice": Lattice,
    "stream": Stream,
    "network": Network,
    "hybrid": Hybrid,
}


class PDFHandler:
    """Handles all operations on the PDF's.

    Handles all operations like temp directory creation, splitting
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

    def __init__(
        self,
        filepath: StrByteType | Path | str,
        pages="1",
        password=None,
        debug=False,
    ):
        self.debug = debug
        if is_url(filepath):
            filepath = download_url(str(filepath))
        self.filepath: StrByteType | Path | str = filepath

        if isinstance(filepath, str) and not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""  # noqa: S105
        else:
            self.password = password
        self.pages = self._get_pages(pages)

    def _get_pages(self, pages):
        """Convert pages string to list of integers.

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

    def _save_page(
        self, filepath: StrByteType | Path, page: int, temp: str, **layout_kwargs
    ) -> tuple[
        Any,
        tuple[float, float],
        list[LTImage],
        list[LTChar],
        list[LTTextLineHorizontal],
        list[LTTextLineVertical],
    ]:
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        temp : str
            Tmp directory.


        Returns
        -------
        layout : object

        dimensions : tuple
            The dimensions of the pdf page

        filepath : str
            The path of the single page PDF - either the original, or a
            normalized version.

        """
        infile = PdfReader(filepath, strict=False)
        if infile.is_encrypted:
            infile.decrypt(self.password)
        fpath = os.path.join(temp, f"page-{page}.pdf")
        froot, fext = os.path.splitext(fpath)
        p = infile.pages[page - 1]
        outfile = PdfWriter()
        outfile.add_page(p)
        with open(fpath, "wb") as f:
            outfile.write(f)
        layout, dimensions = get_page_layout(fpath, **layout_kwargs)
        # fix rotated PDF
        images, chars, horizontal_text, vertical_text = get_image_char_and_text_objects(
            layout
        )
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
            # Only recompute layout and dimension after rotating the pdf
            layout, dimensions = get_page_layout(fpath, **layout_kwargs)
            images, chars, horizontal_text, vertical_text = (
                get_image_char_and_text_objects(layout)
            )
            instream.close()
            return layout, dimensions, images, chars, horizontal_text, vertical_text
        return layout, dimensions, images, chars, horizontal_text, vertical_text

    def parse(
        self,
        flavor: str = "lattice",
        suppress_stdout: bool = False,
        parallel: bool = False,
        layout_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Extract tables by calling parser.get_tables on all single page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use.
            Lattice is used by default.
        suppress_stdout : bool (default: False)
            Suppress logs and warnings.
        parallel : bool (default: False)
            Process pages in parallel using all available cpu cores.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.
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
        # parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
        parser_obj = PARSERS[flavor]
        parser = parser_obj(debug=self.debug, **kwargs)

        with TemporaryDirectory() as tempdir:
            cpu_count = mp.cpu_count()
            # Using multiprocessing only when cpu_count > 1 to prevent a stallness issue
            # when cpu_count is 1
            if parallel and len(self.pages) > 1 and cpu_count > 1:
                with mp.get_context("spawn").Pool(processes=cpu_count) as pool:
                    jobs = []
                    for p in self.pages:
                        j = pool.apply_async(
                            self._parse_page,
                            (p, tempdir, parser, suppress_stdout, layout_kwargs),
                        )
                        jobs.append(j)

                    for j in jobs:
                        t = j.get()
                        tables.extend(t)
            else:
                for p in self.pages:
                    t = self._parse_page(
                        p, tempdir, parser, suppress_stdout, layout_kwargs
                    )
                    tables.extend(t)

        return TableList(sorted(tables))

    def _parse_page(
        self, page: int, tempdir: str, parser, suppress_stdout: bool, layout_kwargs
    ):
        """Extract tables by calling parser.get_tables on a single page PDF.

        Parameters
        ----------
        page : int
            Page number to parse
        parser : Lattice, Stream, Network or Hybrid
            The parser to use.
        suppress_stdout : bool
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        layout, dimensions, images, chars, horizontal_text, vertical_text = (
            self._save_page(self.filepath, page, tempdir, **layout_kwargs)
        )
        page_path = os.path.join(tempdir, f"page-{page}.pdf")
        parser.prepare_page_parse(
            page_path,
            layout,
            dimensions,
            page,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs=layout_kwargs,
        )
        tables = parser.extract_tables()
        return tables
