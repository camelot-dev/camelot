import pytest

from camelot.image_processing import coalesce_collinear_lines


def test_empty_input():
    assert coalesce_collinear_lines([], "horizontal") == []
    assert coalesce_collinear_lines([], "vertical") == []


def test_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        coalesce_collinear_lines([(0, 0, 1, 0)], "diagonal")


def test_horizontal_overlapping_segments_merge():
    # Two segments on the same y that overlap → one spanning segment.
    lines = [(0, 5, 60, 5), (40, 5, 100, 5)]
    out = coalesce_collinear_lines(lines, "horizontal")
    assert out == [(0, 5, 100, 5)]


def test_horizontal_small_gap_within_tol_merges():
    # Gap of 1 (< tol 2) → merge.
    out = coalesce_collinear_lines([(0, 5, 50, 5), (51, 5, 100, 5)], "horizontal")
    assert out == [(0, 5, 100, 5)]


def test_horizontal_large_gap_stays_separate():
    # Gap of 20 (> tol) → two distinct lines.
    out = coalesce_collinear_lines([(0, 5, 50, 5), (70, 5, 100, 5)], "horizontal")
    assert sorted(out) == [(0, 5, 50, 5), (70, 5, 100, 5)]


def test_horizontal_near_equal_y_averaged():
    # y differs by 1 (< tol) → same line, y averaged.
    out = coalesce_collinear_lines([(0, 5, 50, 5), (40, 6, 100, 6)], "horizontal")
    assert len(out) == 1
    x0, y, x1, y2 = out[0]
    assert (x0, x1) == (0, 100) and y == y2 == 5.5


def test_horizontal_distinct_y_stay_separate():
    out = coalesce_collinear_lines([(0, 5, 100, 5), (0, 80, 100, 80)], "horizontal")
    assert len(out) == 2


def test_vertical_overlapping_segments_merge():
    out = coalesce_collinear_lines([(10, 0, 10, 60), (10, 40, 10, 100)], "vertical")
    assert out == [(10, 0, 10, 100)]


def test_vertical_large_gap_stays_separate():
    out = coalesce_collinear_lines([(10, 0, 10, 40), (10, 70, 10, 100)], "vertical")
    assert sorted(out) == [(10, 0, 10, 40), (10, 70, 10, 100)]


def test_chain_of_three_segments_merges_to_one():
    lines = [(0, 5, 30, 5), (30, 5, 60, 5), (60, 5, 90, 5)]
    assert coalesce_collinear_lines(lines, "horizontal") == [(0, 5, 90, 5)]


def test_noop_when_nothing_to_merge():
    lines = [(0, 5, 100, 5)]
    assert coalesce_collinear_lines(lines, "horizontal") == [(0, 5, 100, 5)]
