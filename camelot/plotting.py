"""Plotting functions usefull for visual debugging."""

from pdfminer.layout import LTTextLineVertical


try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True

from .utils import bbox_from_str
from .utils import bbox_from_textlines
from .utils import get_textline_coords


def extend_axe_lim(ax, bbox, margin=10):
    """Ensure the ax limits include the input bbox."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, bbox[0] - margin), max(x1, bbox[2] + margin))
    ax.set_ylim(min(y0, bbox[1] - margin), max(y1, bbox[3] + margin))


def draw_labeled_bbox(
    ax,
    bbox,
    text,
    color="black",
    linewidth=3,
    linestyle="solid",
    label_pos="top,left",
    fontsize=12,
):
    """Utility drawing function to draw a box with an associated text label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        matplotlib.axes.Axes (optional)
    bbox : [type]
        boundingbox
    text : string
        The text to be placed inside the box.
    color : str, optional
        The color of the box, by default "black"
    linewidth : int, optional
        The linewidth of the box, by default 3
    linestyle : str, optional
        The matplotlib linestyle, by default "solid"
    label_pos : str, optional
        The label postiion, by default "top,left"
    fontsize : int, optional
        The fontsize of the text in the box, by default 12
    """
    ax.add_patch(
        patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            fill=False,
        )
    )

    vlabel, hlabel = label_pos.split(",")
    if vlabel == "top":
        y = max(bbox[1], bbox[3])
    elif vlabel == "bottom":
        y = min(bbox[1], bbox[3])
    else:
        y = 0.5 * (bbox[1] + bbox[3])

    # We want to draw the label outside the box (above or below)
    label_align_swap = {"top": "bottom", "bottom": "top", "center": "center"}
    vlabel_out_of_box = label_align_swap[vlabel]
    if hlabel == "right":
        x = max(bbox[0], bbox[2])
    elif hlabel == "left":
        x = min(bbox[0], bbox[2])
    else:
        x = 0.5 * (bbox[0] + bbox[2])
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color="black",
        verticalalignment=vlabel_out_of_box,
        horizontalalignment=hlabel,
        bbox=dict(facecolor=color, alpha=0.1),
    )


def draw_pdf(table, ax):
    """Draw the content of the table's source pdf into the passed subplot.

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)
    """
    img = table.get_pdf_image()
    ax.imshow(img, extent=(0, table.pdf_size[0], 0, table.pdf_size[1]))


def draw_parse_constraints(table, ax):
    """Draw any user provided constraints (area, region, columns, etc).

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)
    """
    if table.parse_details:
        zone_constraints = {
            "region": "table_regions",
            "area": "table_areas",
        }
        for zone_name, zone_id in zone_constraints.items():
            # Display a bbox per region / area
            for zone_str in table.parse_details[zone_id] or []:
                draw_labeled_bbox(
                    ax,
                    bbox_from_str(zone_str),
                    "{zone_name}: ({zone_str})".format(
                        zone_name=zone_name, zone_str=zone_str
                    ),
                    color="purple",
                    linestyle="dotted",
                    linewidth=1,
                    label_pos="bottom,right",
                )


def draw_text(table, ax):
    """Draw text, horizontal in blue, vertical in red.

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)
    """
    bbox = bbox_from_textlines(table.textlines)
    for t in table.textlines:
        color = "red" if isinstance(t, LTTextLineVertical) else "blue"
        ax.add_patch(
            patches.Rectangle(
                (t.x0, t.y0), t.x1 - t.x0, t.y1 - t.y0, color=color, alpha=0.2
            )
        )
    extend_axe_lim(ax, bbox)


def prepare_plot(table, ax=None):
    """Initialize plot and draw common components.

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
    draw_pdf(table, ax)
    draw_parse_constraints(table, ax)
    return ax


class PlotMethods:
    """Classmethod for plotting methods."""

    def __call__(self, table, kind="text", filename=None, ax=None):
        """Plot elements found on PDF page based on kind specified.

        Useful for debugging and playing with different
        parameters to get the best output.

        Parameters
        ----------
        table: camelot.core.Table
            A Camelot Table.
        kind : str, optional (default: 'text')
            {'text', 'grid', 'contour', 'joint', 'line',
                'network_table_search'}
            The element type for which a plot should be generated.
        filename: str, optional (default: None)
            Absolute path for saving the generated plot.
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")

        if table.flavor == "lattice" and kind in ["textedge"]:
            raise NotImplementedError(f"Lattice flavor does not support kind={kind!r}")
        if table.flavor != "lattice" and kind in ["line"]:
            raise NotImplementedError(
                f"{table.flavor} flavor does not support kind={kind!r}"
            )

        plot_method = getattr(self, kind)
        if filename is not None:
            fig = plot_method(table, ax)
            fig.savefig(filename)
            return None

        return plot_method(table, ax)

    def text(self, table, ax=None):
        """Generate a plot for all text elements present on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
        draw_text(table, ax)
        return ax.get_figure()

    @staticmethod
    def grid(table, ax=None):
        """Generate a plot for the detected table grids on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
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
        return ax.get_figure()

    @staticmethod
    def contour(table, ax=None):
        """Generate a plot for all table boundaries present on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        _for_lattice = table.flavor == "lattice"
        ax = prepare_plot(table, ax)

        if not _for_lattice:
            draw_text(table, ax)

        ax.add_patch(
            patches.Rectangle(
                (table._bbox[0], table._bbox[1]),
                table._bbox[2] - table._bbox[0],
                table._bbox[3] - table._bbox[1],
                fill=False,
                color="red",
            )
        )

        if not _for_lattice:
            extend_axe_lim(ax, table._bbox)
        return ax.get_figure()

    @staticmethod
    def textedge(table, ax=None):
        """Generate a plot for relevant textedges.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
        draw_text(table, ax)

        if table.flavor == "network":
            for network in table.parse_details["network_searches"]:
                most_connected_tl = network.most_connected_textline()

                ax.add_patch(
                    patches.Rectangle(
                        (most_connected_tl.x0, most_connected_tl.y0),
                        most_connected_tl.x1 - most_connected_tl.x0,
                        most_connected_tl.y1 - most_connected_tl.y0,
                        color="red",
                        alpha=0.5,
                    )
                )
                for tl in sorted(
                    network._textline_to_alignments.keys(),
                    key=lambda textline: (-textline.y0, textline.x0),
                ):
                    alignments = network._textline_to_alignments[tl]
                    coords = get_textline_coords(tl)
                    alignment_id_h, tls_h = alignments.max_v()
                    alignment_id_v, tls_v = alignments.max_h()
                    xs = list(map(lambda tl: tl.x0, tls_v))
                    ys = list(map(lambda tl: tl.y1, tls_h))
                    top_h = max(ys)
                    ax.text(
                        coords[alignment_id_h],
                        top_h + 5,
                        f"{len(tls_h)}",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=8,
                        color="green",
                    )
                    ax.plot(
                        [coords[alignment_id_h]] * len(ys),
                        ys,
                        color="green",
                        linestyle="solid",
                        linewidth=1,
                        marker="o",
                        markeredgecolor="green",
                        fillstyle=None,
                        markersize=4,
                        alpha=0.8,
                    )

                    left_v = min(map(lambda tl: tl.x0, tls_v))
                    ax.text(
                        left_v - 5,
                        coords[alignment_id_v],
                        f"{len(tls_v)}",
                        verticalalignment="center",
                        horizontalalignment="right",
                        fontsize=8,
                        color="blue",
                    )
                    ax.plot(
                        xs,
                        [coords[alignment_id_v]] * len(xs),
                        color="blue",
                        linestyle="solid",
                        linewidth=1,
                        marker="o",
                        markeredgecolor="blue",
                        fillstyle="full",
                        markersize=3,
                        alpha=0.8,
                    )
        else:
            for te in table._textedges:
                ax.plot([te.coord, te.coord], [te.y0, te.y1])
        return ax.get_figure()

    @staticmethod
    def joint(table, ax=None):
        """Generate a plot for all line intersections present on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
        x_coord = []
        y_coord = []
        for coord in table.parse["joints"]:
            x_coord.append(coord[0])
            y_coord.append(coord[1])
        ax.plot(x_coord, y_coord, "ro")
        return ax.get_figure()

    @staticmethod
    def line(table, ax=None):
        """Generate a plot for all line segments present on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
        vertical, horizontal = table._segments
        for v in vertical:
            ax.plot([v[0], v[2]], [v[1], v[3]])
        for h in horizontal:
            ax.plot([h[0], h[2]], [h[1], h[3]])
        return ax.get_figure()

    @staticmethod
    def network_table_search(table, ax=None):
        """Generate a plot illustrating the steps of the network table search.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        ax = prepare_plot(table, ax)
        if table.parse_details is None:
            return ax.get_figure()
        parse_details = table.parse_details
        for box_id, bbox_search in enumerate(parse_details["bbox_searches"]):
            max_h_gap = bbox_search["max_h_gap"]
            max_v_gap = bbox_search["max_v_gap"]
            iterations = bbox_search["iterations"]
            for iteration, bbox in enumerate(iterations):
                final = iteration == len(iterations) - 1

                draw_labeled_bbox(
                    ax,
                    bbox,
                    f"t{box_id}/i{iteration}",
                    color="red",
                    linewidth=5 if final else 2,
                    fontsize=14 if final else 8,
                    label_pos="bottom,left",
                )

                ax.add_patch(
                    patches.Rectangle(
                        (bbox[0] - max_h_gap, bbox[1] - max_v_gap),
                        bbox[2] - bbox[0] + 2 * max_h_gap,
                        bbox[3] - bbox[1] + 2 * max_v_gap,
                        color="orange",
                        linestyle="dotted",
                        fill=False,
                    )
                )

        for box_id, col_search in enumerate(parse_details["col_searches"]):
            draw_labeled_bbox(
                ax,
                col_search["bbox_full"],
                f"box body + header #{box_id}",
                color="red",
                linewidth=4,
                label_pos="top,left",
            )
            draw_labeled_bbox(
                ax,
                col_search["bbox_body"],
                f"box body #{box_id}",
                color="cyan",
                linewidth=2,
                label_pos="bottom,right",
            )
            for col_anchor in col_search["cols_anchors"]:
                # Display a green line at the col boundary line throughout the
                # table bbox.
                ax.plot(
                    [col_anchor, col_anchor],
                    [
                        col_search["bbox_body"][1],
                        col_search["bbox_body"][3],
                    ],
                    color="green",
                )

        return ax.get_figure()
