"""#35: Network suppresses nested/overlapping duplicate table detections."""

from types import SimpleNamespace

from camelot.parsers import Network
from camelot.parsers.network import _overlap_fraction


def _t(x1, y1, x2, y2):
    return SimpleNamespace(_bbox=(x1, y1, x2, y2))


def test_overlap_fraction_identical():
    assert _overlap_fraction((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_overlap_fraction_disjoint():
    assert _overlap_fraction((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_overlap_fraction_nested():
    # inner fully inside outer -> 1.0
    assert _overlap_fraction((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0


def test_suppresses_nested_duplicate():
    # 'nested' shares the top of 'big' and is fully within its x/y -> dropped.
    big = _t(0, 0, 100, 100)
    nested = _t(0, 40, 100, 100)  # 60% of big's height, fully inside
    out = Network()._postprocess_tables([big, nested])
    assert out == [big]


def test_keeps_disjoint_tables():
    top = _t(0, 60, 100, 100)
    bottom = _t(0, 0, 100, 40)  # no y-overlap
    out = Network()._postprocess_tables([top, bottom])
    assert len(out) == 2


def test_preserves_input_order():
    a = _t(0, 60, 100, 100)
    b = _t(0, 0, 100, 40)
    assert Network()._postprocess_tables([a, b]) == [a, b]


def test_noop_with_single_or_missing_bbox():
    one = _t(0, 0, 10, 10)
    assert Network()._postprocess_tables([one]) == [one]
    no_bbox = [SimpleNamespace(_bbox=None), SimpleNamespace(_bbox=None)]
    assert Network()._postprocess_tables(no_bbox) == no_bbox
