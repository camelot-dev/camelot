"""Functions to handle all operations on the PDF's."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from functools import partial
from itertools import chain
from pathlib import Path
from typing import IO
from typing import Any
from typing import Union

import playa
from playa.exceptions import PDFPasswordIncorrect
from playa.exceptions import PDFTextExtractionNotAllowed
from playa.miner import LTChar
from playa.miner import LTImage
from playa.miner import LTTextLineHorizontal
from playa.miner import LTTextLineVertical

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


FilepathOrBuffer = Union[str, Path, bytes, bytearray, memoryview, IO[bytes]]


def _spill_bytes_to_tempfile(data: bytes) -> str:
    """Write `data` to a NamedTemporaryFile and return its path.

    Used by ``PDFHandler`` when the caller passes a ``bytes``-like object
    or a file-like stream rather than a filesystem path — the Lattice
    flavor's OpenCV image-conversion backend needs a real on-disk file,
    so the simplest contract is "always spill once, treat as a path from
    here on, clean up on close()". The file is created with ``delete=False``
    and reaped in :meth:`PDFHandler.close`.
    """
    with tempfile.NamedTemporaryFile(
        prefix="camelot-", suffix=".pdf", delete=False
    ) as f:
        f.write(data)
        return f.name


class PDFHandler:
    """Handles all operations on the PDF's.

    Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str, Path, bytes, or binary file-like
        Source PDF. Accepts a filesystem path / URL, or — since #270 —
        a ``bytes``-like object or any binary stream with a ``.read()``
        method (``io.BytesIO``, an open ``"rb"`` file, ``requests``
        response ``.raw``, etc). In the in-memory cases the bytes are
        spilled to a temporary file once and cleaned up when the handler
        is closed; this keeps the rest of the pipeline (in particular
        the Lattice OpenCV image-conversion backend) unchanged.
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
        filepath: FilepathOrBuffer,
        pages="1",
        password=None,
        debug=False,
    ):
        self.debug = debug
        self.is_temp_file = False

        if isinstance(filepath, (bytes, bytearray, memoryview)):
            # Raw bytes input — spill to a tempfile and treat as a path.
            self.filepath = _spill_bytes_to_tempfile(bytes(filepath))
            self.is_temp_file = True
        elif hasattr(filepath, "read") and callable(filepath.read):
            # Binary file-like (BytesIO, open('rb', ...), urlopen response,
            # requests Response.raw, etc). Read once, preserve caller's
            # cursor position so they can keep using the stream.
            tell = getattr(filepath, "tell", None)
            seek = getattr(filepath, "seek", None)
            pos = tell() if callable(tell) else None
            data = filepath.read()
            if pos is not None and callable(seek):
                try:
                    seek(pos)
                except (OSError, ValueError):
                    # Non-seekable stream (e.g. a network socket) —
                    # nothing to restore, drop silently.
                    pass
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "file-like 'filepath' must return bytes from .read(),"
                    f" got {type(data).__name__}"
                )
            self.filepath = _spill_bytes_to_tempfile(bytes(data))
            self.is_temp_file = True
        else:
            # Path or URL (existing behaviour).
            self.is_temp_file = is_url(filepath)
            if is_url(filepath):
                filepath = download_url(str(filepath))
            self.filepath = filepath

        if password is None:
            self.password = ""  # noqa: S105
        else:
            self.password = password
        # Defer page resolution until parse() opens the PDF, so that we don't
        # open the document twice per read_pdf call. The literal default
        # "1" doesn't need the PDF at all and is resolved eagerly.
        self._pages_spec = pages
        self._pages_cache: list[int] | None = [1] if pages == "1" else None

    @property
    def pages(self) -> list[int]:
        """Resolved 1-based page numbers, sorted and de-duplicated.

        Lazy: only opens the PDF if the spec is something other than the
        default ``"1"``. Cached after first access.
        """
        if self._pages_cache is None:
            self._pages_cache = self._resolve_pages(self._pages_spec)
        return self._pages_cache

    def _resolve_pages(self, pages: str, pdf: Any | None = None) -> list[int]:
        """Convert the ``pages`` spec to a sorted, de-duplicated list of ints.

        Pass an already-open ``pdf`` to avoid the playa.open() round-trip
        that would otherwise be needed to read ``len(pdf.pages)``.
        """
        if pages == "1":
            return [1]
        if pdf is None:
            with playa.open(self.filepath, space="page", password=self.password) as pdf:
                return self._resolve_pages(pages, pdf)
        return self._expand_pages_spec(pages, len(pdf.pages))

    @staticmethod
    def _expand_pages_spec(pages: str, page_count: int) -> list[int]:
        """Expand a pages spec string (``"all"``, ``"1,3,4"``, ``"1,4-end"``)."""
        page_numbers: list[dict[str, int]] = []
        if pages == "all":
            page_numbers.append({"start": 1, "end": page_count})
        else:
            for r in pages.split(","):
                if "-" in r:
                    a, b = r.split("-")
                    b_int = page_count if b == "end" else int(b)
                    page_numbers.append({"start": int(a), "end": b_int})
                else:
                    page_numbers.append({"start": int(r), "end": int(r)})
        result: list[int] = []
        for p in page_numbers:
            result.extend(range(p["start"], p["end"] + 1))
        return sorted(set(result))

    def __enter__(self) -> PDFHandler:
        """Allow ``PDFHandler`` to be used as a context manager.

        On ``__exit__`` any temp file created by :func:`download_url` (when
        the caller passed a URL) is removed — see :meth:`close`.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the URL-downloaded temp file, if any."""
        self.close()

    def close(self) -> None:
        """Delete the URL-downloaded temp file, if any.

        Idempotent; safe to call from both ``__exit__`` and an explicit
        ``handler.close()`` call. No-op when ``filepath`` was a user-owned
        path (we never delete a file the caller passed in).
        """
        if not self.is_temp_file:
            return
        path = self.filepath
        if isinstance(path, (str, Path)) and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                # On Windows (issue #678) pdfium / playa can still hold
                # an open handle to the temp file when close() runs,
                # giving WinError 32 ("being used by another process").
                # Leave the file behind and let the OS reap it later
                # (NamedTemporaryFile's default tempdir is wiped at
                # reboot) — losing a few KB beats raising mid-cleanup.
                pass
            # Mark cleaned so a second close() doesn't re-stat-and-remove.
            self.is_temp_file = False

    # Kept for backwards-compat with anything that called the old name.
    # New code should access self.pages or call _resolve_pages directly.

    def _get_pages(self, pages):
        """Convert pages string to list of integers.

        .. deprecated::
            Use the :attr:`pages` property; this method exists only for
            backwards-compat with callers that imported it directly.

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
            with playa.open(self.filepath, space="page", password=self.password) as pdf:
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

    def _get_layout(self, page: playa.Page, **layout_kwargs) -> tuple[
        Any,
        tuple[float, float],
        list[LTImage],
        list[LTChar],
        list[LTTextLineHorizontal],
        list[LTTextLineVertical],
        str,
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
            # de-rotate the page
            if rotation == "clockwise":
                # rotate -90 degrees
                page.set_initial_ctm(page.space, page.rotate - 90)
            elif rotation == "anticlockwise":
                # rotate 90 degrees
                page.set_initial_ctm(page.space, page.rotate + 90)
            else:
                raise AssertionError(
                    f"rotation should be clockwise or anticlockwise, is {rotation}"
                )
            # now re-run layout analysis
            layout, dimensions = get_page_layout(page, **layout_kwargs)
            images, chars, horizontal_text, vertical_text = (
                get_image_char_and_text_objects(layout)
            )
        return (
            layout,
            dimensions,
            images,
            chars,
            horizontal_text,
            vertical_text,
            rotation,
        )

    def parse(
        self,
        flavor: str = "lattice",
        suppress_stdout: bool = False,
        parallel: bool = False,
        cpu_count: int | None = None,
        layout_kwargs: dict[str, Any] | None = None,
        per_page: dict[int, dict[str, Any]] | None = None,
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
        cpu_count : int, optional (default: None)
            Maximum number of worker processes to use when ``parallel`` is
            True. ``None`` (default) uses all available cores. Values are
            clamped to ``[1, multiprocessing.cpu_count()]``. Ignored when
            ``parallel`` is False.
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
        per_page = per_page or {}

        # Default parser used by any page without a per_page override.
        parser_obj = PARSERS[flavor]
        parser = parser_obj(debug=self.debug, **kwargs)

        # Compute worker count up-front so we can pass it to playa.open(). The
        # old code also gated the parallel branch on len(self.pages) > 1, but
        # touching self.pages here would force a separate playa.open() to
        # read the page count *before* this one — exactly the redundant open
        # this change exists to remove. playa.pages[…].map(…) honours
        # max_workers regardless of page count, and a single-page doc just
        # uses one worker effectively.
        max_cpus = mp.cpu_count()
        if parallel and max_cpus > 1:
            cpu_count = (
                max_cpus if cpu_count is None else max(1, min(cpu_count, max_cpus))
            )
        else:
            cpu_count = 1
        try:
            with playa.open(
                self.filepath,
                password=self.password,
                space="page",
                max_workers=cpu_count,
            ) as pdf:
                if not pdf.is_extractable:
                    raise PDFTextExtractionNotAllowed(
                        f"Text extraction is not allowed: {self.filepath}"
                    )
                # Resolve pages using the already-open document instead of
                # opening it a second time via the .pages property.
                if self._pages_cache is None:
                    self._pages_cache = self._resolve_pages(self._pages_spec, pdf)
                pages = [x - 1 for x in self._pages_cache]
                tables = pdf.pages[pages].map(
                    partial(
                        self._parse_page,
                        parser=parser,
                        layout_kwargs=layout_kwargs,
                        flavor=flavor,
                        base_kwargs=kwargs,
                        per_page=per_page,
                    )
                )
        except PDFPasswordIncorrect as e:
            raise RuntimeError("File has not been decrypted") from e
        except PDFTextExtractionNotAllowed:
            raise
        return TableList(sorted(chain.from_iterable(tables)))

    def _parse_page(
        self,
        page: playa.Page,
        parser,
        layout_kwargs,
        flavor: str = "lattice",
        base_kwargs: dict[str, Any] | None = None,
        per_page: dict[int, dict[str, Any]] | None = None,
    ):
        """Extract tables by calling parser.get_tables on a single page PDF.

        Parameters
        ----------
        page : playa.Page
            Page to parse
        parser : Lattice, Stream, Network or Hybrid
            The default parser to use when no per-page override applies.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.
        flavor : str, optional
            The global flavor; used as the fallback when a per-page override
            doesn't itself supply ``flavor=``.
        base_kwargs : dict, optional
            The global (already-cleaned) parser kwargs. Merged with any
            per-page override to construct a fresh parser for that page.
        per_page : dict, optional
            Page-number-keyed kwargs overrides (already validated upstream).

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        # playa's page_idx is 0-indexed; user-facing per_page uses 1-indexed.
        page_no = page.page_idx + 1
        overrides = (per_page or {}).get(page_no)
        if overrides:
            page_flavor = overrides.get("flavor", flavor)
            merged = dict(base_kwargs or {})
            for k, v in overrides.items():
                if k != "flavor":
                    merged[k] = v
            page_parser = PARSERS[page_flavor](debug=self.debug, **merged)
        else:
            page_parser = parser
        layout, dimensions, images, chars, horizontal_text, vertical_text, rotation = (
            self._get_layout(page, **layout_kwargs)
        )
        page_parser.prepare_page_parse(
            self.filepath,
            layout,
            dimensions,
            page.page_idx + 1,
            images,
            horizontal_text,
            vertical_text,
            rotation,
            layout_kwargs=layout_kwargs,
        )
        tables = page_parser.extract_tables()
        return tables
