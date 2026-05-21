"""Vector-native contour/joint detection — Stage 2 of #763.

Pure-geometry tests for find_contours_from_lines / find_joints_from_lines.
Lines are synthetic (x0, y0, x1, y1) tuples in PDF coords — no PDF parse,
matching the shape find_lines_from_layout returns.

Convention: horizontal lines are (x0, y, x1, y); vertical lines are
(x, y0, x, y1). PDF coords are y-up.
"""

from camelot.image_processing import _line_crossing
from camelot.image_processing import find_contours_from_lines
from camelot.image_processing import find_joints_from_lines


def _grid(x_left, x_right, y_bot, y_top, xs, ys):
    """Build a ruled grid of horizontal + vertical lines.

    Horizontal lines at each y in ``ys`` span ``[x_left, x_right]``;
    vertical lines at each x in ``xs`` span ``[y_bot, y_top]``.
    """
    h = [(x_left, y, x_right, y) for y in ys]
    v = [(x, y_bot, x, y_top) for x in xs]
    return h, v


# --- _line_crossing -------------------------------------------------------


def test_crossing_basic():
    h = (0, 100, 200, 100)
    v = (50, 0, 50, 200)
    assert _line_crossing(h, v) == (50, 100)


def test_no_crossing_out_of_span():
    h = (0, 100, 40, 100)  # h only spans x 0..40
    v = (50, 0, 50, 200)  # v at x=50 — outside h's x-span
    assert _line_crossing(h, v) is None


def test_crossing_within_tolerance():
    # v stops 1 unit short of h's left edge — still crosses within tol=2.
    h = (10, 100, 200, 100)
    v = (9, 0, 9, 200)
    assert _line_crossing(h, v, tol=2.0) == (9, 100)


def test_crossing_reversed_endpoints():
    # endpoints given right-to-left / top-to-bottom still detected.
    h = (200, 100, 0, 100)
    v = (50, 200, 50, 0)
    assert _line_crossing(h, v) == (50, 100)


# --- find_contours_from_lines --------------------------------------------


def test_empty_inputs_no_contours():
    assert find_contours_from_lines([], []) == []
    assert find_contours_from_lines([(0, 1, 9, 1)], []) == []


def test_single_grid_one_contour():
    # 3x3 ruled grid -> 9 joints (> _MIN_JOINTS=4) -> one table.
    h, v = _grid(0, 200, 0, 200, xs=[0, 100, 200], ys=[0, 100, 200])
    contours = find_contours_from_lines(h, v)
    assert len(contours) == 1
    x0, y0, x1, y1 = contours[0]
    assert (x0, y0, x1, y1) == (0, 0, 200, 200)


def test_two_separate_grids_two_contours():
    # The Stage-1 prototype merged these; connected-components must not.
    h1, v1 = _grid(0, 100, 300, 400, xs=[0, 50, 100], ys=[300, 350, 400])
    h2, v2 = _grid(0, 100, 0, 100, xs=[0, 50, 100], ys=[0, 50, 100])
    contours = find_contours_from_lines(h1 + h2, v1 + v2)
    assert len(contours) == 2
    # sorted top-first: the y=300..400 grid before the y=0..100 grid.
    assert contours[0][1] == 300
    assert contours[1][1] == 0


def test_sparse_lines_below_joint_threshold_dropped():
    # A single cross = 1 joint, well under _MIN_JOINTS -> no table.
    h = [(0, 50, 100, 50)]
    v = [(50, 0, 50, 100)]
    assert find_contours_from_lines(h, v) == []


def test_l_corner_not_a_table():
    # Two lines meeting at one corner -> 1 joint -> dropped.
    h = [(0, 0, 100, 0)]
    v = [(0, 0, 0, 100)]
    assert find_contours_from_lines(h, v) == []


# --- find_joints_from_lines ----------------------------------------------


def test_joints_single_grid():
    h, v = _grid(0, 200, 0, 200, xs=[0, 100, 200], ys=[0, 100, 200])
    joints = find_joints_from_lines(h, v)
    assert len(joints) == 1
    ((bbox, pts),) = joints.items()
    assert bbox == (0, 0, 200, 200)
    # 3 h-lines x 3 v-lines = 9 intersection points.
    assert len(pts) == 9
    assert (100, 100) in pts  # centre joint
    assert (0, 0) in pts and (200, 200) in pts  # corners


def test_joints_two_grids_keyed_separately():
    h1, v1 = _grid(0, 100, 300, 400, xs=[0, 50, 100], ys=[300, 350, 400])
    h2, v2 = _grid(0, 100, 0, 100, xs=[0, 50, 100], ys=[0, 50, 100])
    joints = find_joints_from_lines(h1 + h2, v1 + v2)
    assert len(joints) == 2
    for bbox, pts in joints.items():
        assert len(pts) == 9  # each grid is fully crossed
        # every joint lies within its own bbox
        x0, y0, x1, y1 = bbox
        for px, py in pts:
            assert x0 <= px <= x1 and y0 <= py <= y1


def test_joints_partial_grid_within_tolerance():
    # vertical lines stop 1 unit short of the top horizontal line; tol bridges.
    h = [(0, 0, 100, 0), (0, 50, 100, 50), (0, 100, 100, 100)]
    v = [(0, 0, 0, 99), (50, 0, 50, 99), (100, 0, 100, 99)]
    joints = find_joints_from_lines(h, v, tol=2.0)
    assert len(joints) == 1
    ((_, pts),) = joints.items()
    assert len(pts) == 9  # tol=2 lets the y=100 row still register
