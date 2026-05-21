"""engine= kwarg + layout_has_ruled_lines probe — #763 Stage 2b plumbing."""

import pytest

from camelot.image_processing import layout_has_ruled_lines


class _MockLTLine:
    def __init__(self, x0, y0, x1, y1, stroke=True):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.stroke = stroke

    def __iter__(self):
        return iter(())


class _MockContainer:
    def __init__(self, objs):
        self._objs = objs

    def __iter__(self):
        return iter(self._objs)


def test_probe_none_layout_is_false():
    assert layout_has_ruled_lines(None) is False


def test_probe_below_threshold_false(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)
    # 3 lines, default min_lines=4 -> False
    layout = _MockContainer([_MockLTLine(0, i, 10, i) for i in range(3)])
    assert layout_has_ruled_lines(layout) is False


def test_probe_at_threshold_true(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)
    layout = _MockContainer([_MockLTLine(0, i, 10, i) for i in range(4)])
    assert layout_has_ruled_lines(layout) is True


def test_probe_custom_threshold(monkeypatch):
    from camelot import image_processing as ip

    monkeypatch.setattr(ip, "LTLine", _MockLTLine)
    monkeypatch.setattr(ip, "LTContainer", _MockContainer)
    layout = _MockContainer([_MockLTLine(0, i, 10, i) for i in range(2)])
    assert layout_has_ruled_lines(layout, min_lines=2) is True
    assert layout_has_ruled_lines(layout, min_lines=3) is False


def test_lattice_engine_validation():
    from camelot.parsers.lattice import Lattice

    # Valid values construct fine.
    for eng in ("raster", "vector", "combined", "auto"):
        assert Lattice(engine=eng).engine == eng
    # Invalid value rejected at construction.
    with pytest.raises(ValueError, match="engine must be"):
        Lattice(engine="opencv")


def test_lattice_default_engine_is_raster():
    from camelot.parsers.lattice import Lattice

    assert Lattice().engine == "raster"


def test_vector_engine_extracts_without_rendering(foo_pdf):
    """engine='vector' parses foo.pdf and never rasterises the page."""
    import camelot

    tables = camelot.read_pdf(foo_pdf, flavor="lattice", engine="vector")
    assert len(tables) == 1
    assert tables[0].shape[0] >= 2 and tables[0].shape[1] >= 2


def test_vector_engine_matches_raster_on_vector_ruled_pdf(foo_pdf):
    """Vector output equals raster on a crisp vector-ruled PDF.

    foo.pdf's rules are real vector strokes, so detecting from them
    directly (no render) should reconstruct the same grid the raster
    pipeline finds — the strict oracle for the render-free path.
    """
    import camelot

    raster = camelot.read_pdf(foo_pdf, flavor="lattice", engine="raster")
    vector = camelot.read_pdf(foo_pdf, flavor="lattice", engine="vector")
    assert len(vector) == len(raster) == 1
    assert vector[0].shape == raster[0].shape
    assert vector[0].df.equals(raster[0].df)


def test_auto_engine_runs(foo_pdf):
    """engine='auto' must not raise — resolves to combined/raster (#763)."""
    import camelot

    tables = camelot.read_pdf(foo_pdf, flavor="lattice", engine="auto")
    assert len(tables) == 1


def test_combined_engine_matches_raster_on_vector_ruled_pdf(foo_pdf):
    """Combined output equals raster on a crisp vector-ruled PDF.

    foo.pdf's ruled lines are crisp vector strokes that the raster
    engine already finds, so unioning the same vector lines in must not
    change the result — this is the strict oracle for the 'combined'
    integration: if the PDF->image transform or mask drawing were wrong,
    the cell grid (and thus the DataFrame) would differ.
    """
    import camelot

    raster = camelot.read_pdf(foo_pdf, flavor="lattice", engine="raster")
    combined = camelot.read_pdf(foo_pdf, flavor="lattice", engine="combined")
    assert len(combined) == len(raster) == 1
    assert combined[0].df.equals(raster[0].df)
    assert combined[0].shape == raster[0].shape


def test_combined_engine_through_hybrid(foo_pdf):
    """flavor='hybrid' forwards engine='combined' to its lattice half."""
    import camelot

    tables = camelot.read_pdf(foo_pdf, flavor="hybrid", engine="combined")
    assert len(tables) >= 1


def test_hybrid_forwards_engine_to_lattice():
    """Hybrid constructs its Lattice sub-parser with the requested engine."""
    from camelot.parsers.hybrid import Hybrid

    h = Hybrid(engine="combined")
    assert h.lattice_parser.engine == "combined"
    # default stays raster
    assert Hybrid().lattice_parser.engine == "raster"
