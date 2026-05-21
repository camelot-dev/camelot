"""copy_text correctly fills cells spanned in both directions (#349)."""

from camelot.core import Table


def _build_table(width=3, height=3):
    """Make a width x height grid of cells with neutral default flags.

    ``Table.__init__`` expects ``cols`` / ``rows`` as lists of
    (start, end) tuples, not flat coordinate lists — build them
    explicitly so ``Cell(c[0], r[1], c[1], r[0])`` works.
    """
    xs = [0, 100, 200, 300][: width + 1]
    ys = [300, 200, 100, 0][: height + 1]
    cols = list(zip(xs[:-1], xs[1:], strict=True))
    rows = list(zip(ys[:-1], ys[1:], strict=True))
    table = Table(cols, rows)
    # Cells already initialised by Table; explicitly set every flag False.
    for row in table.cells:
        for cell in row:
            cell.left = cell.right = cell.top = cell.bottom = True
            cell.text = ""
    return table


def test_copy_text_horizontal_only_unchanged():
    table = _build_table()
    table.cells[0][0].text = "header"
    # Mark the top row as horizontally-spanning so copies should propagate.
    table.cells[0][1].left = False
    table.cells[0][2].left = False
    table.copy_spanning_text(copy_text=["h"])
    assert table.cells[0][1].text == "header"
    assert table.cells[0][2].text == "header"


def test_copy_text_vertical_only_unchanged():
    table = _build_table()
    table.cells[0][0].text = "category"
    # Mark left column as vertically-spanning.
    table.cells[1][0].top = False
    table.cells[2][0].top = False
    table.copy_spanning_text(copy_text=["v"])
    assert table.cells[1][0].text == "category"
    assert table.cells[2][0].text == "category"


def test_copy_text_2d_span_filled():
    """A 2x2 block spanning both directions should fully propagate (#349).

    Without the iterative loop, the bottom-right cell of a 2x2 spanned
    block stays empty because its source cell (top-right OR bottom-left)
    hasn't been filled yet during the single-pass copy.
    """
    table = _build_table(width=3, height=3)
    table.cells[0][0].text = "X"
    # Top-right of the 2x2 span — needs horizontal copy from [0][0].
    table.cells[0][1].left = False
    # Bottom-left of the 2x2 span — needs vertical copy from [0][0].
    table.cells[1][0].top = False
    # Bottom-right of the 2x2 span — needs BOTH horizontal AND vertical.
    table.cells[1][1].left = False
    table.cells[1][1].top = False

    table.copy_spanning_text(copy_text=["h", "v"])
    assert table.cells[0][1].text == "X"
    assert table.cells[1][0].text == "X"
    assert table.cells[1][1].text == "X", "2D-spanned cell stays empty (#349)"


def test_copy_text_2d_span_reverse_order():
    """Same span, asking for ['v', 'h'] — still fills the 2D-spanned cell."""
    table = _build_table(width=3, height=3)
    table.cells[0][0].text = "Y"
    table.cells[0][1].left = False
    table.cells[1][0].top = False
    table.cells[1][1].left = False
    table.cells[1][1].top = False

    table.copy_spanning_text(copy_text=["v", "h"])
    assert table.cells[1][1].text == "Y"


def test_copy_text_none_is_noop():
    table = _build_table()
    table.cells[0][0].text = "keep"
    table.copy_spanning_text(copy_text=None)
    assert table.cells[0][0].text == "keep"
    assert table.cells[1][0].text == ""
