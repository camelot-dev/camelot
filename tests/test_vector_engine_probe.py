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
    for eng in ("raster", "vector", "auto"):
        assert Lattice(engine=eng).engine == eng
    # Invalid value rejected at construction.
    with pytest.raises(ValueError, match="engine must be"):
        Lattice(engine="opencv")


def test_lattice_default_engine_is_raster():
    from camelot.parsers.lattice import Lattice

    assert Lattice().engine == "raster"


def test_explicit_vector_engine_not_yet_implemented(testdir, foo_pdf):
    """engine='vector' surfaces a clear NotImplementedError pointing at #763."""
    import camelot

    with pytest.raises(NotImplementedError, match="#763"):
        camelot.read_pdf(foo_pdf, flavor="lattice", engine="vector")


def test_auto_engine_falls_back_to_raster(foo_pdf):
    """engine='auto' must not raise — it resolves to raster until 2b lands."""
    import camelot

    tables = camelot.read_pdf(foo_pdf, flavor="lattice", engine="auto")
    assert len(tables) == 1
