"""Unit tests for the ``show`` parameter of ``PlotMethods.__call__`` (issue #296).

``camelot.plot(table).show()`` calls matplotlib's ``Figure.show()`` which
returns immediately in non-interactive (script) backends, so the window
closes before the user can see it.

The fix adds ``show=True`` to ``camelot.plot()`` which internally calls
``plt.show(block=True)`` — the correct blocking entry point for scripts.
These tests verify that the parameter is wired correctly without needing
a real PDF or a display.
"""

from unittest.mock import MagicMock, patch, call
import pytest


def _make_table(flavor="lattice"):
    """Return a minimal mock that satisfies PlotMethods.__call__."""
    table = MagicMock()
    table.flavor = flavor
    return table


@pytest.fixture()
def plot_methods():
    from camelot.plotting import PlotMethods

    return PlotMethods()


class TestPlotMethodsShowParameter:
    """Regression tests for the show=True blocking fix (issue #296)."""

    def test_show_false_does_not_call_plt_show(self, plot_methods):
        """Default (show=False) must not call plt.show() — existing behaviour unchanged."""
        table = _make_table()
        mock_fig = MagicMock()

        with patch.object(plot_methods, "text", return_value=mock_fig) as mock_plot, \
             patch("camelot.plotting.plt.show") as mock_plt_show:
            plot_methods(table, kind="text", show=False)

        mock_plt_show.assert_not_called()

    def test_show_true_calls_plt_show_with_block(self, plot_methods):
        """show=True must call plt.show(block=True) so scripts stay open."""
        table = _make_table()
        mock_fig = MagicMock()

        with patch.object(plot_methods, "text", return_value=mock_fig), \
             patch("camelot.plotting.plt.show") as mock_plt_show:
            result = plot_methods(table, kind="text", show=True)

        mock_plt_show.assert_called_once_with(block=True)
        assert result is mock_fig

    def test_show_true_returns_figure(self, plot_methods):
        """show=True must still return the figure so callers can further inspect it."""
        table = _make_table()
        sentinel_fig = MagicMock(name="sentinel_fig")

        with patch.object(plot_methods, "text", return_value=sentinel_fig), \
             patch("camelot.plotting.plt.show"):
            result = plot_methods(table, kind="text", show=True)

        assert result is sentinel_fig

    def test_show_ignored_when_filename_given(self, plot_methods):
        """When filename is provided the figure is saved, not displayed."""
        table = _make_table()
        mock_fig = MagicMock()

        with patch.object(plot_methods, "text", return_value=mock_fig), \
             patch("camelot.plotting.plt.show") as mock_plt_show:
            result = plot_methods(table, kind="text", filename="/tmp/out.png", show=True)

        mock_plt_show.assert_not_called()
        mock_fig.savefig.assert_called_once_with("/tmp/out.png")
        assert result is None
