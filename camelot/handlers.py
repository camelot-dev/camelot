"""Functions to handle all operations on the PDF's."""

from __future__ import annotations

import multiprocessing as mp
import os
from itertools import chain
from functools import partial
from pathlib import Path
from typing import Any

import playa
import paves.miner as pm
from playa.utils import mult_matrix
from paves.miner import LTChar
from paves.miner import LTImage
from paves.miner import LTTextLineHorizontal
from paves.miner import LTTextLineVertical

from .core import TableList
from .parsers import Hybrid
from .parsers import Lattice
from .parsers import Network
from .parsers import Stream
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
        filepath: bytes | Path | str,
        pages="1",
        password=None,
        debug=False,
    ):
        self.debug = debug
        if is_url(filepath):
            filepath = download_url(str(filepath))
        self.filepath: bytes | Path | str = filepath

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
            with playa.open(self.filepath, password=self.password) as pdf:
                page_count = len(pdf.pages)
            if pages == "all":
                page_numbers.append({"start": 1, "end": page_count})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = page_count
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})
        result = []
        for p in page_numbers:
            result.extend(range(p["start"], p["end"] + 1))
        return sorted(set(result))

    def _get_layout(
        self, page: playa.Page, **layout_kwargs
    ) -> tuple[
        Any,
        tuple[float, float],
        list[LTImage],
        list[LTChar],
        list[LTTextLineHorizontal],
        list[LTTextLineVertical],
    ]:
        """Get layout from a page.

        Parameters
        ----------
        page : playa.Page
            Page in the document.


        Returns
        -------
        layout : object

        dimensions : tuple
            The dimensions of the pdf page

        filepath : str
            The path of the single page PDF - either the original, or a
            normalized version.

        """
        layout, dimensions = get_page_layout(page, **layout_kwargs)
        # fix rotated PDF
        images, chars, horizontal_text, vertical_text = get_image_char_and_text_objects(
            layout
        )
        rotation = get_rotation(chars, horizontal_text, vertical_text)
        if rotation:
            width, height = dimensions
            # rotate the page by simply frobbing the CTM
            if rotation == "clockwise":
                page.ctm = mult_matrix((0, 1, -1, 0, height, 0), page.ctm)
            elif rotation == "anticlockwise":
                page.ctm = mult_matrix((0, -1, 1, 0, 0, width), page.ctm)
            else:
                raise AssertionError(
                    f"rotation should be clockwise or anticlockwise, is {rotation}")
            # now re-run layout analysis
            layout, dimensions = get_page_layout(page, **layout_kwargs)
            images, chars, horizontal_text, vertical_text = (
                get_image_char_and_text_objects(layout)
            )
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

        with playa.open(
            self.filepath,
            password=self.password,
            space="page",
        ) as pdf:
            pages = [x - 1 for x in self.pages]
            tables = pdf.pages[pages].map(
                partial(self._parse_page, parser=parser, layout_kwargs=layout_kwargs))
        return TableList(sorted(chain.from_iterable(tables)))

    def _parse_page(
        self, page: playa.Page, parser, layout_kwargs
    ):
        """Extract tables by calling parser.get_tables on a single page PDF.

        Parameters
        ----------
        page : playa.Page
            Page to parse
        parser : Lattice, Stream, Network or Hybrid
            The parser to use.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        layout, dimensions, images, chars, horizontal_text, vertical_text = (
            self._get_layout(page, **layout_kwargs)
        )
        parser.prepare_page_parse(
            self.filepath,
            layout,
            dimensions,
            page.page_idx + 1,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs=layout_kwargs,
        )
        tables = parser.extract_tables()
        return tables
