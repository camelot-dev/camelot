# -*- coding: utf-8 -*-

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True


def draw_pdf(table, ax, to_pdf_scale=True):
    """Draw the content of the table's source pdf into the passed subplot

    Parameters
    ----------
    table : camelot.core.Table

    fig : matplotlib.axes.Axes

    """
    img = table.get_pdf_image()
    if to_pdf_scale:
        ax.imshow(img, extent=(0, table.pdf_size[0], 0, table.pdf_size[1]))
    else:
        ax.imshow(img)


class PlotMethods(object):
    def __call__(self, table, kind="text", filename=None):
        """Plot elements found on PDF page based on kind
        specified, useful for debugging and playing with different
        parameters to get the best output.

        Parameters
        ----------
        table: camelot.core.Table
            A Camelot Table.
        kind : str, optional (default: 'text')
            {'text', 'grid', 'contour', 'joint', 'line'}
            The element type for which a plot should be generated.
        filepath: str, optional (default: None)
            Absolute path for saving the generated plot.

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")

        if table.flavor == "lattice" and kind in ["textedge"]:
            raise NotImplementedError(
                "Lattice flavor does not support kind='{}'".format(kind)
            )
        elif table.flavor in ["stream", "hybrid"] and kind in ["line"]:
            raise NotImplementedError(
                "Stream flavor does not support kind='{}'".format(kind)
            )

        plot_method = getattr(self, kind)
        return plot_method(table)

    @staticmethod
    def text(table):
        """Generates a plot for all text elements present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        draw_pdf(table, ax)
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(
                patches.Rectangle(
                        (t[0], t[1]),
                        t[2] - t[0],
                        t[3] - t[1],
                        alpha=0.5
                    )
                )
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)
        return fig

    @staticmethod
    def grid(table):
        """Generates a plot for the detected table grids
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        draw_pdf(table, ax)
        for row in table.cells:
            for cell in row:
                if cell.left:
                    ax.plot([cell.lb[0], cell.lt[0]], [cell.lb[1], cell.lt[1]])
                if cell.right:
                    ax.plot([cell.rb[0], cell.rt[0]], [cell.rb[1], cell.rt[1]])
                if cell.top:
                    ax.plot([cell.lt[0], cell.rt[0]], [cell.lt[1], cell.rt[1]])
                if cell.bottom:
                    ax.plot([cell.lb[0], cell.rb[0]], [cell.lb[1], cell.rb[1]])
        return fig

    @staticmethod
    def contour(table):
        """Generates a plot for all table boundaries present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        _FOR_LATTICE = table.flavor == "lattice"
        draw_pdf(table, ax, to_pdf_scale=not _FOR_LATTICE)

        if _FOR_LATTICE:
            table_bbox = table._bbox_unscaled
        else:
            table_bbox = {table._bbox: None}

        xs, ys = [], []
        if not _FOR_LATTICE:
            for t in table._text:
                xs.extend([t[0], t[2]])
                ys.extend([t[1], t[3]])
                ax.add_patch(
                    patches.Rectangle(
                        (t[0], t[1]), t[2] - t[0], t[3] - t[1], color="blue"
                    )
                )

        for t in table_bbox.keys():
            ax.add_patch(
                patches.Rectangle(
                    (t[0], t[1]), t[2] - t[0], t[3] - t[1],
                    fill=False, color="red"
                )
            )
            if not _FOR_LATTICE:
                xs.extend([t[0], t[2]])
                ys.extend([t[1], t[3]])
                ax.set_xlim(min(xs) - 10, max(xs) + 10)
                ax.set_ylim(min(ys) - 10, max(ys) + 10)
        return fig

    @staticmethod
    def textedge(table):
        """Generates a plot for relevant textedges.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        draw_pdf(table, ax)
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(
                patches.Rectangle(
                    (t[0], t[1]), t[2] - t[0], t[3] - t[1],
                    color="blue",
                    alpha=0.5
                )
            )
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)

        if table.flavor == "hybrid":
            # FRHTODO: Clean this up
            table.debug_info["edges_searches"][0].plot_alignments(ax)
        else:
            for te in table._textedges:
                ax.plot([te.x, te.x], [te.y0, te.y1])
        return fig

    @staticmethod
    def joint(table):
        """Generates a plot for all line intersections present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        draw_pdf(table, ax, to_pdf_scale=False)
        table_bbox = table._bbox_unscaled
        x_coord = []
        y_coord = []
        for k in table_bbox.keys():
            for coord in table_bbox[k]:
                x_coord.append(coord[0])
                y_coord.append(coord[1])
        ax.plot(x_coord, y_coord, "ro")
        return fig

    @staticmethod
    def line(table):
        """Generates a plot for all line segments present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        draw_pdf(table, ax)
        vertical, horizontal = table._segments
        for v in vertical:
            ax.plot([v[0], v[2]], [v[1], v[3]])
        for h in horizontal:
            ax.plot([h[0], h[2]], [h[1], h[3]])
        return fig
