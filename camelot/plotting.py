# -*- coding: utf-8 -*-

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True


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
                f"Lattice flavor does not support kind='{kind}'"
            )
        elif table.flavor == "stream" and kind in ["joint", "line"]:
            raise NotImplementedError(
                f"Stream flavor does not support kind='{kind}'"
            )

        plot_method = getattr(self, kind)
        return plot_method(table)

    def text(self, table):
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
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(patches.Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1]))
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)
        return fig

    def grid(self, table):
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

    def contour(self, table):
        """Generates a plot for all table boundaries present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        try:
            img, table_bbox = table._image
            _FOR_LATTICE = True
        except TypeError:
            img, table_bbox = (None, {table._bbox: None})
            _FOR_LATTICE = False
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")

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
                    (t[0], t[1]), t[2] - t[0], t[3] - t[1], fill=False, color="red"
                )
            )
            if not _FOR_LATTICE:
                xs.extend([t[0], t[2]])
                ys.extend([t[1], t[3]])
                ax.set_xlim(min(xs) - 10, max(xs) + 10)
                ax.set_ylim(min(ys) - 10, max(ys) + 10)

        if _FOR_LATTICE:
            ax.imshow(img)
        return fig

    def textedge(self, table):
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
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(
                patches.Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1], color="blue")
            )
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)

        for te in table._textedges:
            ax.plot([te.x, te.x], [te.y0, te.y1])

        return fig

    def joint(self, table):
        """Generates a plot for all line intersections present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        img, table_bbox = table._image
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
        x_coord = []
        y_coord = []
        for k in table_bbox.keys():
            for coord in table_bbox[k]:
                x_coord.append(coord[0])
                y_coord.append(coord[1])
        ax.plot(x_coord, y_coord, "ro")
        ax.imshow(img)
        return fig

    def line(self, table):
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
        vertical, horizontal = table._segments
        for v in vertical:
            ax.plot([v[0], v[2]], [v[1], v[3]])
        for h in horizontal:
            ax.plot([h[0], h[2]], [h[1], h[3]])
        return fig
