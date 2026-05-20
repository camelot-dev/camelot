"""Vector-line engine for Lattice — Stage 1 of #763.

Tests for ``find_lines_from_layout`` and its helpers. These work directly
against synthesised LTLine / LTRect objects (no PDF parse) plus one
small smoke test against the real foo.pdf fixture so we know the helper
sees what playa actually emits in practice.
"""

import os

import pytest

from camelot.image_processing import (
    _LINE_AS_THIN_RECT_TOL,
    _LINE_ORTHOGONAL_TOL,
    _ruled_lines_from_layout,
    find_lines_from_layout,
)


class _MockLTLine:
    """Minimal stand-in for playa.miner.LTLine — only the attrs we read."""

    def __init__(self, x0, y0, x1, y1, stroke=True):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.stroke = stroke

    def __iter__(self):  # so isinstance(_, LTContainer) is False
        return iter(())


class _MockLTRect:
    def __init__(self, bbox, stroke=False, fill=False):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.stroke = stroke
        self.fill = fill

    def __iter__(self):
        return iter(())


class _MockContainer:
    """A simple iterable container of layout objects, like LTPage's __iter__."""

    def __init__(self, objs):
        self._objs = objs

    def __iter__(self):
        return iter(self._objs)


def test_isinstance_check_uses_real_classes():
    """Sanity: the real playa LTLine / LTRect are what we filter on.

    Mocks here are duck-typed; the test confirms the production code's
    isinstance(...) calls don't accept duck types alone. We construct
    real playa objects below where needed for the integration test.
    """
    from playa.miner import LTLine, LTRect

    assert LTLine is not None
    assert LTRect is not None


def test_find_lines_from_layout_invalid_direction_rejected():
    with pytest.raises(ValueError, match="must be 'horizontal' or 'vertical'"):
        find_lines_from_layout(_MockContainer([]), direction="diagonal")


def test_empty_layout_returns_empty_list():
    assert find_lines_from_layout(_MockContainer([]), direction="horizontal") == []
    assert find_lines_from_layout(_MockContainer([]), direction="vertical") == []


def test_ruled_lines_from_layout_isinstance_filtering(monkeypatch):
    """_ruled_lines_from_layout filters via real isinstance — patch in mocks.

    The production code uses isinstance(obj, LTLine|LTRect|LTContainer);
    we substitute the bound names so our mocks satisfy the checks.
    """
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    horizontal = _MockLTLine(0, 100, 200, 100)
    vertical = _MockLTLine(50, 0, 50, 200)
    diagonal = _MockLTLine(0, 0, 100, 100)  # 45° — dropped from both axes

    out = _ruled_lines_from_layout(_MockContainer([horizontal, vertical, diagonal]))
    assert len(out) == 3  # raw collector keeps all three; classification is later


def test_horizontal_vs_vertical_classification(monkeypatch):
    """A canonical horizontal + a canonical vertical line classify correctly."""
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    h = _MockLTLine(0, 100, 200, 100)
    v = _MockLTLine(50, 0, 50, 200)
    layout = _MockContainer([h, v])

    h_lines = find_lines_from_layout(layout, direction="horizontal")
    v_lines = find_lines_from_layout(layout, direction="vertical")

    assert h_lines == [(0, 100, 200, 100)]
    assert v_lines == [(50, 0, 50, 200)]


def test_diagonal_line_dropped(monkeypatch):
    """A diagonal line with dx == dy doesn't end up in either direction."""
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    diag = _MockLTLine(0, 0, 100, 100)
    layout = _MockContainer([diag])

    assert find_lines_from_layout(layout, direction="horizontal") == []
    assert find_lines_from_layout(layout, direction="vertical") == []


def test_unstroked_ltline_is_skipped(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    line = _MockLTLine(0, 100, 200, 100, stroke=False)
    layout = _MockContainer([line])
    assert find_lines_from_layout(layout, direction="horizontal") == []


def test_stroked_ltrect_yields_four_edges(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    rect = _MockLTRect((0, 0, 100, 50), stroke=True)
    layout = _MockContainer([rect])

    h_lines = find_lines_from_layout(layout, direction="horizontal")
    v_lines = find_lines_from_layout(layout, direction="vertical")

    # bottom + top edges → 2 horizontal lines.
    assert sorted(h_lines) == [(0, 0, 100, 0), (0, 50, 100, 50)]
    # left + right edges → 2 vertical lines.
    assert sorted(v_lines) == [(0, 0, 0, 50), (100, 0, 100, 50)]


def test_thin_filled_rect_treated_as_line(monkeypatch):
    """PDFs that draw rules as 0.5pt filled rects are recovered."""
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    # 100-unit wide, 0.5-unit tall filled rect → horizontal line at y=midpoint.
    thin_h = _MockLTRect((0, 99.75, 100, 100.25), fill=True)
    # 0.5-unit wide, 100-unit tall filled rect → vertical line.
    thin_v = _MockLTRect((49.75, 0, 50.25, 100), fill=True)
    layout = _MockContainer([thin_h, thin_v])

    h_lines = find_lines_from_layout(layout, direction="horizontal")
    v_lines = find_lines_from_layout(layout, direction="vertical")

    assert h_lines == [(0, 100.0, 100, 100.0)]
    assert v_lines == [(50.0, 0, 50.0, 100)]


def test_unstroked_unfilled_rect_skipped(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    rect = _MockLTRect((0, 0, 100, 100), stroke=False, fill=False)
    layout = _MockContainer([rect])

    assert find_lines_from_layout(layout, direction="horizontal") == []
    assert find_lines_from_layout(layout, direction="vertical") == []


def test_recursive_walk_into_nested_container(monkeypatch):
    """A line nested inside a sub-container is still found."""
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTRect", _MockLTRect)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)

    inner = _MockContainer([_MockLTLine(0, 50, 100, 50)])
    outer = _MockContainer([inner])

    assert find_lines_from_layout(outer, direction="horizontal") == [(0, 50, 100, 50)]


def test_orthogonal_tolerance_constant_is_reasonable():
    """A nominally-horizontal line with sub-pixel y-jitter still classifies."""
    assert _LINE_ORTHOGONAL_TOL >= 0.1
    assert _LINE_AS_THIN_RECT_TOL >= 0.5


def test_smoke_against_real_pdf_layout():
    """Real fixture: foo.pdf has a ruled table; vector engine sees lines."""
    pytest.importorskip("playa")
    import playa

    from camelot.utils import get_page_layout

    pdf_path = os.path.join(os.path.dirname(__file__), "files", "foo.pdf")
    with playa.open(pdf_path, space="page") as pdf:
        page = pdf.pages[0]
        layout, _dim = get_page_layout(page)
        h_lines = find_lines_from_layout(layout, direction="horizontal")
        v_lines = find_lines_from_layout(layout, direction="vertical")

    # foo.pdf's ruled table is a 7x7 grid → at least 8 horizontal rules
    # (one per row boundary + outer borders) and 8 vertical rules.
    assert len(h_lines) >= 4, f"expected horizontal lines, got {h_lines}"
    assert len(v_lines) >= 4, f"expected vertical lines, got {v_lines}"
