"""IO related functions to Read the PDF and returns extracted tables."""

import os
import warnings
from pathlib import Path
from typing import Any

from .handlers import PDFHandler
from .utils import TemporaryDirectory
from .utils import remove_extra
from .utils import validate_input

# Minimum vertical + horizontal segment count for the auto-flavor heuristic
# to call a page "ruled". Two lines per axis catches even tiny ruled tables
# (a 2-row 2-col grid produces 3 horizontal + 3 vertical lines including
# borders) while keeping borderless pages with one stray underline accent
# from being mis-classified.
_AUTO_FLAVOR_LINE_THRESHOLD = 2


def _detect_flavor(filepath, password=None):
    """Pick the most appropriate flavor for a PDF.

    Renders the first page, thresholds it, and counts ruled horizontal and
    vertical line segments. Used when the caller passes ``flavor="auto"``.

    Returns
    -------
    str
        Either ``"lattice"`` (enough ruled lines on the rendered page) or
        ``"network"`` (else). ``"network"`` is also the fallback when
        rendering itself fails (e.g. unreadable PDF, missing backend
        dependencies) — the assumption is that giving the text-based parser
        a chance is more useful than raising before parsing starts.
    """
    # Local imports keep \`camelot.read_pdf\` import-time cheap — cv2/playa
    # imports already weigh in for the parsers; deferring these for the
    # default \`flavor="lattice"\` path costs nothing.
    from .backends import ImageConversionBackend
    from .image_processing import adaptive_threshold
    from .image_processing import find_lines

    try:
        backend = ImageConversionBackend()
        with TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "auto_flavor_probe.png")
            backend.convert(str(filepath), png_path, resolution=300, page=1)
            if not os.path.exists(png_path):
                return "network"
            _, threshold = adaptive_threshold(png_path, process_background=False)
            # Use the Lattice default line_scale (15) — picking 40 here
            # excludes legitimate small/medium ruled tables. The same
            # value the lattice parser itself uses by default.
            _, v_segments = find_lines(threshold, direction="vertical", line_scale=15)
            _, h_segments = find_lines(threshold, direction="horizontal", line_scale=15)
    except Exception:
        # Any failure on the probe (no usable backend, encrypted page, broken
        # PDF, OpenCV import surprise) is *not* fatal — the user asked us to
        # pick, we pick the more forgiving option and let the parser report
        # the real error if any.
        return "network"

    has_grid = (
        len(v_segments) >= _AUTO_FLAVOR_LINE_THRESHOLD
        and len(h_segments) >= _AUTO_FLAVOR_LINE_THRESHOLD
    )
    return "lattice" if has_grid else "network"


def _normalize_per_page(per_page):
    """Coerce ``per_page`` to ``{int: dict}`` form, raising ValueError on bad input.

    Accepts None / empty (returns ``{}``), int or str keys, and dict
    values. Other shapes raise ``ValueError`` with a precise message
    naming the offending entry. Values are shallow-copied so a later
    in-place edit doesn't mutate the caller's dict.
    """
    if per_page is None:
        return {}
    per_page_norm: dict[int, dict[str, Any]] = {}
    for k, v in per_page.items():
        try:
            page_no = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"per_page keys must be page numbers, got {k!r}") from exc
        if not isinstance(v, dict):
            raise ValueError(
                f"per_page[{k!r}] must be a dict of kwargs," f" got {type(v).__name__}"
            )
        per_page_norm[page_no] = dict(v)
    return per_page_norm


def _validate_per_page(per_page_norm, global_flavor):
    """Validate each per-page override against its effective flavor.

    Each entry's flavor is either an explicit override (must be one of
    the four concrete flavors — ``"auto"`` doesn't make sense
    page-by-page) or the global flavor. All non-flavor kwargs are then
    checked with the existing :func:`validate_input` against that
    effective flavor.
    """
    for page_no, overrides in per_page_norm.items():
        page_flavor = overrides.get("flavor", global_flavor)
        if page_flavor not in ("lattice", "stream", "network", "hybrid"):
            raise NotImplementedError(
                f"per_page[{page_no}] flavor={page_flavor!r} is not"
                " one of: 'lattice', 'stream', 'network', 'hybrid'."
                " ('auto' is only valid as the global flavor.)"
            )
        page_kwargs = {k: v for k, v in overrides.items() if k != "flavor"}
        validate_input(page_kwargs, flavor=page_flavor)


def read_pdf(
    filepath: str | Path,
    pages="1",
    password=None,
    flavor="lattice",
    suppress_stdout=False,
    parallel=False,
    cpu_count=None,
    layout_kwargs=None,
    per_page=None,
    debug=False,
    **kwargs,
):
    """Read PDF and return extracted tables.

    Note: kwargs annotated with ^ can only be used with flavor='stream' or flavor='network'
    and kwargs annotated with * can only be used with flavor='lattice'.
    The hybrid parser accepts kwargs with both annotations.

    Parameters
    ----------
    filepath : str, Path, IO
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    flavor : str (default: 'lattice')
        The parsing method to use. Valid values:

        - ``'lattice'`` (default): line-ruled tables.
        - ``'stream'``: borderless tables with whitespace-separated columns.
        - ``'network'``: borderless tables via text-edge alignment connectivity.
        - ``'hybrid'``: combines layout- and image-based analysis.
        - ``'auto'``: render the first requested page, count ruled
          horizontal/vertical lines, and pick ``lattice`` when both axes
          show at least 2 segments; pick ``network`` otherwise. A
          ``UserWarning`` is emitted telling you which flavor was selected.
    suppress_stdout : bool, optional (default: False)
        Print all logs and warnings.
    parallel : bool, optional (default: False)
        Process pages in parallel using all available cpu cores.
    cpu_count : int, optional (default: None)
        Maximum number of worker processes when ``parallel=True``. ``None``
        (default) uses all available cores. Values are clamped to
        ``[1, multiprocessing.cpu_count()]``. Ignored when
        ``parallel=False``.
    layout_kwargs : dict, optional (default: {})
        A dict of `pdfminer.layout.LAParams
        <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.
    per_page : dict, optional (default: None)
        Per-page parameter overrides. Maps a 1-indexed page number (int
        or str) to a dict of any keyword argument otherwise valid for
        ``read_pdf``. Values supplied here override the globally-supplied
        kwargs for that one page only — every other page keeps the global
        values. Useful for multi-layout PDFs where different pages need
        different ``table_areas``, ``columns``, ``flavor``, etc. The
        per-page ``flavor`` itself may be overridden; the global flavor
        applies otherwise. Originally proposed by @sverma25 in #41.

        Example::

            tables = camelot.read_pdf(
                "report.pdf",
                pages="1-3",
                flavor="stream",
                split_text=True,
                per_page={2: {"table_areas": ["120, 210, 400, 90"]}},
            )

        Here pages 1 and 3 use the global ``flavor="stream", split_text=True``
        only; page 2 uses both *and* the page-specific ``table_areas``.
    table_areas : list, optional (default: None)
        List of table area strings of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    columns^ : list, optional (default: None)
        List of column x-coordinates strings where the coordinates
        are comma-separated.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str or sequence of str, optional (default: '')
        Characters or substrings to strip from each cell before
        assignment. A ``str`` strips per-character — every character in
        the string is removed wherever it appears (e.g. ``" \\n"`` drops
        all spaces and newlines). A list/tuple of ``str`` strips whole
        substrings (e.g. ``["[1]", "[2]"]`` removes those footnote
        markers but leaves bare ``[``/``]`` alone). Whole-substring
        mode requested in #484.
    row_tol^ : int, optional (default: 2)
        Tolerance parameter used to combine text vertically,
        to generate rows.
    column_tol^ : int, optional (default: 0)
        Tolerance parameter used to combine text horizontally,
        to generate columns.
    process_background* : bool, optional (default: False)
        Process background lines.
    line_scale* : int, optional (default: 15)
        Line size scaling factor. The larger the value the smaller
        the detected lines. Making it very large will lead to text
        being detected as lines.
    copy_text* : list, optional (default: None)
        {'h', 'v'}
        Direction in which text in a spanning cell will be copied
        over.
    shift_text* : list, optional (default: ['l', 't'])
        {'l', 'r', 't', 'b'}
        Direction in which text in a spanning cell will flow.
    line_tol* : int, optional (default: 2)
        Tolerance parameter used to merge close vertical and horizontal
        lines.
    joint_tol* : int, optional (default: 2)
        Tolerance parameter used to decide whether the detected lines
        and points lie close to each other.
    threshold_blocksize* : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    threshold_constant* : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    iterations* : int, optional (default: 0)
        Number of times for erosion/dilation is applied.

        For more information, refer `OpenCV's dilate
        <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.
    backend* : str, optional by default "pdfium"
        The backend to use for converting the PDF to an image so it can be processed by OpenCV.
    use_fallback* : bool, optional
        Fallback to another backend if unavailable, by default True
    resolution* : int, optional (default: 300)
        Resolution used for PDF to PNG conversion.

    Returns
    -------
    tables : camelot.core.TableList

    Notes
    -----
    **Encrypted PDFs / extraction permissions** (#590). Camelot honours the
    ``/Encrypt`` dictionary's text-extraction permission: ``read_pdf`` raises
    :class:`playa.exceptions.PDFTextExtractionNotAllowed` if the PDF is
    encrypted and the user-password permission set forbids text extraction.
    The check fires on the document object returned by ``playa.open`` while
    the encryption metadata is still attached — this is a real behavioural
    change vs the pre-1.0 backend, where per-page temp-PDF splitting
    silently dropped the metadata so the check was effectively a no-op.
    Note: PDF spec only enforces the flag through the encryption layer —
    for **unencrypted** PDFs that carry a "no extraction" claim via
    ``/Perms``, there is no enforcement mechanism and Camelot extracts.
    Supplying the document owner password through ``password=`` bypasses
    the user-password permission set (matches every other PDF tool).

    """
    if layout_kwargs is None:
        layout_kwargs = {}
    if flavor not in ["lattice", "stream", "network", "hybrid", "auto"]:
        raise NotImplementedError(
            "Unknown flavor specified."
            " Use either 'lattice', 'stream', 'network', 'hybrid' or 'auto'"
        )

    per_page_norm = _normalize_per_page(per_page)

    with warnings.catch_warnings():
        if suppress_stdout:
            warnings.simplefilter("ignore")

        with PDFHandler(filepath, pages=pages, password=password, debug=debug) as p:
            if flavor == "auto":
                flavor = _detect_flavor(p.filepath, password=p.password or None)
                warnings.warn(
                    f"camelot.read_pdf: auto-detected flavor={flavor!r}",
                    UserWarning,
                    stacklevel=2,
                )
            validate_input(kwargs, flavor=flavor)
            kwargs = remove_extra(kwargs, flavor=flavor)

            _validate_per_page(per_page_norm, flavor)

            tables = p.parse(
                flavor=flavor,
                suppress_stdout=suppress_stdout,
                parallel=parallel,
                cpu_count=cpu_count,
                layout_kwargs=layout_kwargs,
                per_page=per_page_norm,
                **kwargs,
            )
        return tables
