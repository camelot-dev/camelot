"""Implementation of the ML (Table Transformer) table parser.

The ``flavor='ml'`` backend follows one rule: **the model supplies the
structure, the PDF supplies the text.**

A neural table-structure model (Microsoft's Table Transformer / TATR, loaded
from HuggingFace) is run on the rendered page to detect rows, columns and
spanning cells *as boxes*. Those boxes are turned into the same
column/row/spanning grid the heuristic parsers build, and every cell's text is
then filled from the PDF's own text layer via :func:`get_table_index` — exactly
like the other flavors. The model never emits a single character of cell text,
so it cannot hallucinate or alter a value; OCR-grade substitution errors are
impossible because no recognition of glyphs happens here at all.

The heavy dependencies (``torch`` / ``transformers`` / ``timm``) are optional
and imported lazily — install them with ``pip install 'camelot-py[ml]'``. The
box-to-grid post-processing and the image→PDF coordinate mapping are pure
functions (no torch), so they are unit-tested without the models present.

Text source seam
----------------
Cell text comes from :meth:`MachineLearning._text_source`, which today returns
the born-digital PDF text layer. The structure half already runs on the page
*image*, so a future OCR text source (scanned / image-only PDFs) only has to
provide ``{text fragments with bboxes}`` to the same fill path — see the
roadmap in the ``[ml]`` tier task.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field

from ..backends import ImageConversionBackend
from ..utils import bbox_from_str
from ..utils import scale
from ..utils import scale_pdf
from ..utils import text_in_bbox_per_axis
from ..utils import translate
from .base import BaseParser

#: TATR structure-model class labels we consume.
LABEL_ROW = "table row"
LABEL_COLUMN = "table column"
LABEL_SPANNING = "table spanning cell"

#: Default HuggingFace checkpoints (MIT-licensed). The v1.1-all structure model
#: is trained on PubTables-1M + FinTabNet.c (stronger on financial/borderless).
DEFAULT_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
DEFAULT_DETECTION_MODEL = "microsoft/table-transformer-detection"

#: Two detections whose centres on an axis lie within this many pixels are
#: treated as the same row/column band and merged (the model occasionally emits
#: near-duplicate boxes for one row/column).
DEFAULT_MERGE_TOL = 6.0

#: A spanning box must cover at least this fraction of a grid cell's width/height
#: on an axis for that cell to count as part of the span.
_SPAN_OVERLAP_FRACTION = 0.5


@dataclass
class DetectedObject:
    """One box emitted by the structure model.

    Coordinates are in image-pixel space with a top-left origin (y grows
    downward), matching what ``transformers`` object-detection
    post-processing returns.

    Parameters
    ----------
    label : str
        Class name, e.g. ``"table row"`` / ``"table column"``.
    score : float
        Detection confidence in ``[0, 1]``.
    bbox : tuple
        ``(x0, y0, x1, y1)`` in image pixels.
    """

    label: str
    score: float
    bbox: tuple[float, float, float, float]


@dataclass
class MLGrid:
    """A reconstructed table grid, in image-pixel space.

    Attributes
    ----------
    col_bounds : list
        ``(x_left, x_right)`` per column, left-to-right.
    row_bounds : list
        ``(y_top, y_bottom)`` per row, top-to-bottom (image y grows down).
    spans : list
        ``(r0, c0, r1, c1)`` inclusive grid-index ranges for spanning cells.
    """

    col_bounds: list[tuple[float, float]] = field(default_factory=list)
    row_bounds: list[tuple[float, float]] = field(default_factory=list)
    spans: list[tuple[int, int, int, int]] = field(default_factory=list)


def _axis_intervals(boxes, axis, tol):
    """Collapse boxes to merged ``(lo, hi)`` intervals along one axis.

    Parameters
    ----------
    boxes : list of DetectedObject
    axis : int
        ``0`` for the x-axis (columns), ``1`` for the y-axis (rows).
    tol : float
        Centre-distance under which adjacent intervals are merged.

    Returns
    -------
    list of tuple
        Merged ``(lo, hi)`` intervals, sorted by centre.
    """
    lo_i, hi_i = (0, 2) if axis == 0 else (1, 3)
    raw = sorted(
        ((b.bbox[lo_i], b.bbox[hi_i]) for b in boxes),
        key=lambda t: (t[0] + t[1]) / 2.0,
    )
    merged: list[tuple[float, float]] = []
    for lo, hi in raw:
        if merged:
            prev_centre = (merged[-1][0] + merged[-1][1]) / 2.0
            if abs((lo + hi) / 2.0 - prev_centre) <= tol:
                merged[-1] = (min(merged[-1][0], lo), max(merged[-1][1], hi))
                continue
        merged.append((lo, hi))
    return merged


def _bounds_from_intervals(intervals):
    """Turn merged band intervals into gap-free, ordered cell bounds.

    Separators are placed at the midpoint of each adjacent gap; the outer
    edges keep the first/last interval's extent. The result tiles the axis
    with no gaps or overlaps — the shape :class:`camelot.core.Table` expects.
    """
    if not intervals:
        return []
    seps = [intervals[0][0]]
    for i in range(len(intervals) - 1):
        seps.append((intervals[i][1] + intervals[i + 1][0]) / 2.0)
    seps.append(intervals[-1][1])
    return [(seps[i], seps[i + 1]) for i in range(len(seps) - 1)]


def _interval_overlap(a0, a1, b0, b1, frac):
    """True if ``[b0, b1]`` covers at least ``frac`` of cell ``[a0, a1]``."""
    lo, hi = max(a0, b0), min(a1, b1)
    if hi <= lo:
        return False
    cell = a1 - a0
    return cell <= 0 or (hi - lo) / cell >= frac


def _spans_to_cells(spans, col_bounds, row_bounds):
    """Map spanning boxes to inclusive ``(r0, c0, r1, c1)`` grid ranges.

    A spanning box that resolves to a single cell (covers one row and one
    column) is dropped — it adds no merge information.
    """
    out = []
    for s in spans:
        sx0, sy0, sx1, sy1 = s.bbox
        cols_hit = [
            i
            for i, (a, b) in enumerate(col_bounds)
            if _interval_overlap(a, b, sx0, sx1, _SPAN_OVERLAP_FRACTION)
        ]
        rows_hit = [
            i
            for i, (a, b) in enumerate(row_bounds)
            if _interval_overlap(a, b, sy0, sy1, _SPAN_OVERLAP_FRACTION)
        ]
        if not cols_hit or not rows_hit:
            continue
        r0, r1 = min(rows_hit), max(rows_hit)
        c0, c1 = min(cols_hit), max(cols_hit)
        if r1 > r0 or c1 > c0:
            out.append((r0, c0, r1, c1))
    return out


def objects_to_grid(objects, score_thresh=0.5, merge_tol=DEFAULT_MERGE_TOL):
    """Reconstruct a table grid from structure-model detections (pure).

    Parameters
    ----------
    objects : list of DetectedObject
        Row / column / spanning-cell boxes in image-pixel space.
    score_thresh : float, optional (default: 0.5)
        Drop detections below this confidence.
    merge_tol : float, optional
        Centre-distance under which duplicate row/column bands merge.

    Returns
    -------
    MLGrid
        Column/row bounds (image space) and spanning-cell index ranges.
        Empty bounds mean no usable grid was found.
    """
    kept = [o for o in objects if o.score >= score_thresh]
    cols = [o for o in kept if o.label == LABEL_COLUMN]
    rows = [o for o in kept if o.label == LABEL_ROW]
    spans = [o for o in kept if o.label == LABEL_SPANNING]
    if not cols or not rows:
        return MLGrid()
    col_bounds = _bounds_from_intervals(_axis_intervals(cols, 0, merge_tol))
    row_bounds = _bounds_from_intervals(_axis_intervals(rows, 1, merge_tol))
    span_cells = _spans_to_cells(spans, col_bounds, row_bounds)
    return MLGrid(col_bounds=col_bounds, row_bounds=row_bounds, spans=span_cells)


def grid_to_pdf_cols_rows(grid, pdf_scalers):
    """Map an image-space grid to PDF-space ``cols`` / ``rows`` (pure).

    Mirrors :func:`camelot.utils.scale_image`'s per-coordinate transform,
    including the y-flip (image origin top-left → PDF origin bottom-left).

    Parameters
    ----------
    grid : MLGrid
    pdf_scalers : tuple
        ``(pdf_width_scaler, pdf_height_scaler, image_height)`` — same shape
        Lattice builds for :func:`scale_image`.

    Returns
    -------
    tuple
        ``(cols, rows)`` where ``cols`` is a list of ``(x_left, x_right)``
        increasing and ``rows`` a list of ``(y_top, y_bottom)`` decreasing —
        the order :class:`camelot.core.Table` requires.
    """
    sx, sy, img_h = pdf_scalers
    cols = [(scale(c0, sx), scale(c1, sx)) for c0, c1 in grid.col_bounds]
    rows = []
    for top_img, bot_img in grid.row_bounds:
        y_top = scale(abs(translate(-img_h, top_img)), sy)
        y_bottom = scale(abs(translate(-img_h, bot_img)), sy)
        rows.append((y_top, y_bottom))
    return cols, rows


def apply_spans(table, spans):
    """Open interior edges of each span block so its cells merge (pure).

    Starts from a fully-edged grid (every cell bounded on all sides) and, for
    each spanning block, drops the edges *interior* to the block. A cell with
    an open interior edge reports ``hspan`` / ``vspan`` True, so text flows to
    the block's anchor cell via :meth:`MachineLearning._reduce_index` and
    ``copy_text`` works — the same spanning model Lattice uses.
    """
    for r0, c0, r1, c1 in spans:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cell = table.cells[r][c]
                if c > c0:
                    cell.left = False
                if c < c1:
                    cell.right = False
                if r > r0:
                    cell.top = False
                if r < r1:
                    cell.bottom = False


@dataclass
class _LoadedModels:
    """Holds the lazily-imported torch handle and the two TATR models."""

    torch: object
    det_processor: object
    det_model: object
    str_processor: object
    str_model: object


#: Process-level cache of loaded models, keyed by
#: ``(detection_model, structure_model, device)``. ``read_pdf`` builds a fresh
#: parser per call, so without this every call (and every PDF in a benchmark)
#: would reload the checkpoints. Lives for the life of the process.
_MODEL_CACHE: dict[tuple[str, str, str], _LoadedModels] = {}

#: Process-level OCR engine singleton (lazy). Building a RapidOCR instance
#: loads ONNX models, so reuse it across pages/calls. ``list[object]`` because
#: the concrete RapidOCR type is only importable with the [ocr] extra.
_OCR_ENGINE: list[object] = []


class _OCRWord:
    """A textline-like wrapper around one OCR word, in PDF coordinate space.

    Exposes just what :func:`camelot.utils.get_table_index` /
    :func:`text_in_bbox_per_axis` need (``x0/y0/y1/x1`` with a bottom-left
    origin and ``get_text``). ``split_text`` / ``flag_size`` are *not*
    supported through OCR (they need per-glyph data), so no ``_objs``.
    """

    def __init__(self, text, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self._text = text

    def get_text(self):
        """Return the recognised word text."""
        return self._text


class MachineLearning(BaseParser):
    """Neural table-structure parser (Table Transformer / TATR).

    The model detects table structure on the rendered page; cell text is
    filled from the PDF's text layer. Requires the optional ML dependencies
    (``pip install 'camelot-py[ml]'``).

    Parameters
    ----------
    table_regions : list, optional (default: None)
        Page regions to restrict detection to, ``"x1,y1,x2,y2"`` in PDF
        coordinate space (left-top, right-bottom).
    table_areas : list, optional (default: None)
        Exact table areas, ``"x1,y1,x2,y2"`` in PDF coordinate space. When
        given, table *detection* is skipped and structure recognition runs
        directly on each area.
    copy_text : list, optional (default: None)
        ``{'h', 'v'}`` — directions to copy text across spanning cells.
    shift_text : list, optional (default: ['l', 't'])
        ``{'l', 'r', 't', 'b'}`` — direction text in a spanning cell flows.
    split_text : bool, optional (default: False)
        Split text that spans multiple cells.
    flag_size : bool, optional (default: False)
        Wrap font-size-flagged text in ``<s></s>``.
    strip_text : str, optional (default: '')
        Characters stripped from a cell before assignment.
    structure_model : str, optional
        HuggingFace checkpoint for structure recognition.
    detection_model : str, optional
        HuggingFace checkpoint for table detection.
    device : str, optional (default: 'cpu')
        Torch device, e.g. ``'cpu'`` or ``'cuda'``.
    detection_threshold : float, optional (default: 0.5)
        Score threshold for the detection model.
    structure_threshold : float, optional (default: 0.5)
        Score threshold for the structure model.
    crop_padding : int, optional (default: 10)
        Pixels of margin added around each detected table before structure
        recognition. TATR is trained on padded crops; the margin keeps the
        outermost rows/columns from being clipped.
    ocr : {'auto', True, False}, optional (default: 'auto')
        Where cell text comes from. ``'auto'`` uses the PDF's born-digital
        text layer when present and falls back to OCR on the rendered page
        when the page has none (scanned / image-only PDFs); ``True`` always
        OCRs; ``False`` never does. OCR needs the optional OCR dependencies
        (``pip install 'camelot-py[ocr]'``) and does not support
        ``split_text`` / ``flag_size`` (no per-glyph data).
    resolution : int, optional (default: 300)
        DPI for rendering the page to an image.
    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        copy_text=None,
        shift_text=None,
        split_text=False,
        flag_size=False,
        strip_text="",
        replace_text=None,
        structure_model=DEFAULT_STRUCTURE_MODEL,
        detection_model=DEFAULT_DETECTION_MODEL,
        device="cpu",
        detection_threshold=0.5,
        structure_threshold=0.5,
        crop_padding=10,
        ocr="auto",
        resolution=300,
        use_fallback=True,
        backend="pdfium",
        debug=False,
        **kwargs,
    ):
        super().__init__(
            "machine_learning",
            table_regions=table_regions,
            table_areas=table_areas,
            copy_text=copy_text,
            split_text=split_text,
            strip_text=strip_text,
            replace_text=replace_text,
            shift_text=shift_text or ["l", "t"],
            flag_size=flag_size,
            debug=debug,
        )
        self.structure_model = structure_model
        self.detection_model = detection_model
        self.device = device
        self.detection_threshold = detection_threshold
        self.structure_threshold = structure_threshold
        self.crop_padding = crop_padding
        self.ocr = ocr
        self.resolution = resolution
        self.icb = ImageConversionBackend(use_fallback=use_fallback, backend=backend)
        self._models: _LoadedModels | None = None
        self._ocr_engine = None
        # Per-page render state (set in _render); _ocr_cache reset per page.
        self._image_rgb = None
        self._ocr_cache: list[_OCRWord] | None = None
        self._pdf_scalers: tuple[float, float, float] | None = None
        self._image_scalers: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------ #
    # Lazy model loading (the only torch-dependent state)
    # ------------------------------------------------------------------ #
    def _load_models(self):
        """Import torch/transformers and build the two TATR models once.

        Raises
        ------
        ImportError
            If the optional ML dependencies are not installed, with the
            install hint.
        """
        if self._models is not None:
            return self._models
        cache_key = (self.detection_model, self.structure_model, self.device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._models = cached
            return self._models
        try:
            import torch
            from transformers import AutoImageProcessor
            from transformers import AutoModelForObjectDetection
        except ImportError as exc:  # pragma: no cover - exercised without [ml]
            raise ImportError(
                "flavor='ml' requires the optional ML dependencies. "
                "Install them with: pip install 'camelot-py[ml]'"
            ) from exc

        det_processor = AutoImageProcessor.from_pretrained(self.detection_model)
        det_model = (
            AutoModelForObjectDetection.from_pretrained(self.detection_model)
            .to(self.device)
            .eval()
        )
        str_processor = AutoImageProcessor.from_pretrained(self.structure_model)
        str_model = (
            AutoModelForObjectDetection.from_pretrained(self.structure_model)
            .to(self.device)
            .eval()
        )
        self._normalize_processor_size(det_processor)
        self._normalize_processor_size(str_processor)
        self._models = _LoadedModels(
            torch, det_processor, det_model, str_processor, str_model
        )
        _MODEL_CACHE[cache_key] = self._models
        return self._models

    @staticmethod
    def _normalize_processor_size(processor):
        """Make a ``longest_edge``-only resize config transformers accepts.

        The TATR structure checkpoint ships ``size={'longest_edge': 800}`` (a
        max-longest-edge resize), but DETR's image processor in transformers
        >= 4.x requires ``{height, width}`` or ``{shortest_edge,
        longest_edge}``. Setting both edges to the same value reproduces a
        longest-edge cap exactly — the scale that makes the shortest edge that
        big would push the longest past the cap, so it falls back to capping
        the longest edge, aspect ratio preserved. No-op for valid configs.
        """
        size = getattr(processor, "size", None)
        if isinstance(size, dict) and set(size) == {"longest_edge"}:
            edge = size["longest_edge"]
            processor.size = {"shortest_edge": edge, "longest_edge": edge}

    # ------------------------------------------------------------------ #
    # Text source seam — born-digital today, OCR-pluggable later
    # ------------------------------------------------------------------ #
    def _text_source(self, bbox):
        """Return ``{direction: [textlines]}`` for the cell-fill pass.

        Two sources behind one seam:

        * **born-digital** — the PDF's own text layer clipped to ``bbox``
          (exact characters, the default).
        * **OCR** — words recognised from the page image (scanned / image-only
          PDFs), used when :meth:`_use_ocr` resolves true. OCR words are
          horizontal-only; the vertical axis is empty.
        """
        if self._use_ocr():
            return text_in_bbox_per_axis(bbox, self._ocr_words(), [])
        return text_in_bbox_per_axis(bbox, self.horizontal_text, self.vertical_text)

    def _use_ocr(self) -> bool:
        """Resolve whether this page's cells are filled from OCR.

        ``ocr=True`` always; ``ocr='auto'`` only when the page has no
        born-digital text layer (the scanned case); ``ocr=False`` never.
        """
        if self.ocr is True:
            return True
        if self.ocr == "auto":
            return not self.horizontal_text
        return False

    def _document_has_no_text(self):
        """Don't bail on a text-less page when OCR can supply the text.

        BaseParser short-circuits ``extract_tables`` when the page has no
        text layer; with OCR enabled a scanned page is exactly what we want
        to process, so override that for the OCR case.
        """
        if self._use_ocr():
            return False
        return super()._document_has_no_text()

    def _get_ocr_engine(self):
        """Lazily build (and process-cache) the OCR engine."""
        if self._ocr_engine is not None:
            return self._ocr_engine
        if _OCR_ENGINE:
            self._ocr_engine = _OCR_ENGINE[0]
            return self._ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise ImportError(
                "flavor='ml' with OCR (scanned PDFs) requires the optional OCR "
                "dependencies. Install them with: pip install 'camelot-py[ocr]'"
            ) from exc
        self._ocr_engine = RapidOCR()
        _OCR_ENGINE.append(self._ocr_engine)
        return self._ocr_engine

    def _ocr_words(self):
        """OCR the page image once; return words as PDF-coord :class:`_OCRWord`s."""
        if self._ocr_cache is not None:
            return self._ocr_cache
        import numpy as np

        engine = self._get_ocr_engine()
        result, _ = engine(np.asarray(self._image_rgb))
        words: list[_OCRWord] = []
        sx, sy, img_h = self._pdf_scalers
        for box, text, _score in result or []:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            # image px (top-left origin) -> PDF coords (bottom-left), y flipped.
            x0 = scale(min(xs), sx)
            x1 = scale(max(xs), sx)
            y_top = scale(abs(translate(-img_h, min(ys))), sy)
            y_bottom = scale(abs(translate(-img_h, max(ys))), sy)
            words.append(_OCRWord(text, x0, y_bottom, x1, y_top))
        self._ocr_cache = words
        return words

    # ------------------------------------------------------------------ #
    # Rendering + coordinate scalers (torch-free)
    # ------------------------------------------------------------------ #
    def _render(self):
        """Render the page to an RGB image and build image↔PDF scalers."""
        import cv2
        from PIL import Image

        image_bgr = self.icb.to_array(self.filename, self.page)
        img_h, img_w = image_bgr.shape[0], image_bgr.shape[1]
        self._image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self._ocr_cache = None  # OCR is per-page; clear any previous page's words
        self._pdf_scalers = (
            self.pdf_width / float(img_w),
            self.pdf_height / float(img_h),
            img_h,
        )
        self._image_scalers = (
            img_w / float(self.pdf_width),
            img_h / float(self.pdf_height),
            self.pdf_height,
        )

    # ------------------------------------------------------------------ #
    # Model inference (torch — validated with [ml] installed, step 2)
    # ------------------------------------------------------------------ #
    def _infer(self, image, processor, model, threshold):
        """Run one object-detection model and return DetectedObjects.

        Coordinates are in ``image``'s own pixel space (top-left origin).
        """
        models = self._models
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with models.torch.no_grad():
            outputs = model(**inputs)
        target = models.torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target
        )[0]
        id2label = model.config.id2label
        objects = []
        for sc, lbl, box in zip(  # noqa: B905 - torch tensors, no strict=
            results["scores"], results["labels"], results["boxes"]
        ):
            name = id2label[int(lbl)]
            x0, y0, x1, y1 = (float(v) for v in box)
            objects.append(DetectedObject(name, float(sc), (x0, y0, x1, y1)))
        return objects

    def _detect_table_regions(self):
        """Return table bounding boxes in image-pixel space.

        Uses the detection model unless the user pinned ``table_areas``
        (then those PDF boxes are mapped straight to image space).
        """
        if self.table_areas is not None:
            return [
                scale_pdf(bbox_from_str(area), self._image_scalers)
                for area in self.table_areas
            ]
        objects = self._infer(
            self._image_rgb,
            self._models.det_processor,
            self._models.det_model,
            self.detection_threshold,
        )
        return [o.bbox for o in objects if o.label.startswith("table")]

    def _recognize_structure(self, region):
        """Run structure recognition on one table crop.

        The crop is padded by ``crop_padding`` px (clamped to the image) —
        TATR's structure model was trained on padded table crops, and without
        the margin the outermost rows/columns get clipped. Returns
        DetectedObjects in *full-page* image coordinates (the padded-crop
        offset is added back), so downstream geometry is page-consistent.
        """
        pad = self.crop_padding
        width, height = self._image_rgb.size
        x0 = max(0, int(region[0]) - pad)
        y0 = max(0, int(region[1]) - pad)
        x1 = min(width, int(region[2]) + pad)
        y1 = min(height, int(region[3]) + pad)
        crop = self._image_rgb.crop((x0, y0, x1, y1))
        objects = self._infer(
            crop,
            self._models.str_processor,
            self._models.str_model,
            self.structure_threshold,
        )
        return [
            DetectedObject(
                o.label,
                o.score,
                (o.bbox[0] + x0, o.bbox[1] + y0, o.bbox[2] + x0, o.bbox[3] + y0),
            )
            for o in objects
        ]

    # ------------------------------------------------------------------ #
    # BaseParser pipeline hooks
    # ------------------------------------------------------------------ #
    def _generate_table_bbox(self):
        """Detect tables, recognise structure, store each grid in PDF coords."""
        self._load_models()
        self._render()
        if self._use_ocr() and (self.split_text or self.flag_size):
            # OCR words carry no per-glyph data, so split_text / flag_size
            # can't apply; disable them for this page rather than crash.
            warnings.warn(
                "flavor='ml' OCR mode does not support split_text / flag_size "
                "(no per-glyph data); ignoring them for this page.",
                stacklevel=2,
            )
            self.split_text = False
            self.flag_size = False
        self.table_bbox_parses = {}
        for region in self._detect_table_regions():
            grid = objects_to_grid(
                self._recognize_structure(region),
                score_thresh=self.structure_threshold,
            )
            if not grid.col_bounds or not grid.row_bounds:
                continue
            cols, rows = grid_to_pdf_cols_rows(grid, self._pdf_scalers)
            pdf_bbox = (cols[0][0], rows[-1][1], cols[-1][1], rows[0][0])
            self.table_bbox_parses[pdf_bbox] = {
                "cols": cols,
                "rows": rows,
                "spans": grid.spans,
            }

    def _generate_columns_and_rows(self, bbox, user_cols):
        parse = self.table_bbox_parses[bbox]
        self.t_bbox = self._text_source(bbox)
        # No ruled segments — spanning is carried by parse["spans"], applied
        # in _generate_table; v_s/h_s stay empty.
        return parse["cols"], parse["rows"], [], []

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        table = self._initialize_new_table(table_idx, bbox, cols, rows)
        table.set_all_edges()
        apply_spans(table, self.table_bbox_parses[bbox].get("spans", []))
        self.record_parse_metadata(table)
        return table

    @staticmethod
    def _reduce_index(table, idx, shift_text):
        """Flow text within a spanning cell to its anchor (reuses Lattice)."""
        from .lattice import Lattice

        return Lattice._reduce_index(table, idx, shift_text)

    def _reject_table(self, table) -> bool:
        """Drop empty/degenerate detections (no rows, no cols, all blank)."""
        if table.df.empty or table.shape[0] == 0 or table.shape[1] == 0:
            return True
        return table.whitespace >= 100.0
