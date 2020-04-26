# -*- coding: utf-8 -*-

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True

from .utils import (bbox_from_str, get_textline_coords)


def draw_labeled_bbox(
    ax, bbox, text,
    color="black", linewidth=3,
    linestyle="solid",
    label_pos="top,left"
):
    ax.add_patch(
        patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0], bbox[3] - bbox[1],
            color=color,
            linewidth=linewidth, linestyle=linestyle,
            fill=False
        )
    )

    vlabel, hlabel = label_pos.split(",")
    if (vlabel == "top"):
        y = max(bbox[1], bbox[3])
    elif (vlabel == "bottom"):
        y = min(bbox[1], bbox[3])
    else:
        y = 0.5 * (bbox[1] + bbox[3])

    # We want to draw the label outside the box (above or below)
    label_align_swap = {
        "top": "bottom",
        "bottom": "top",
        "center": "center"
    }
    vlabel_out_of_box = label_align_swap[vlabel]
    if (hlabel == "right"):
        x = max(bbox[0], bbox[2])
    elif (hlabel == "left"):
        x = min(bbox[0], bbox[2])
    else:
        x = 0.5 * (bbox[0] + bbox[2])
    ax.text(
        x, y,
        text,
        fontsize=12, color="black",
        verticalalignment=vlabel_out_of_box,
        horizontalalignment=hlabel,
        bbox=dict(facecolor=color, alpha=0.3)
    )


def draw_pdf(table, ax, to_pdf_scale=True):
    """Draw the content of the table's source pdf into the passed subplot

    Parameters
    ----------
    table : camelot.core.Table

    ax : matplotlib.axes.Axes (optional)

    to_pdf_scale : bool (optional)

    """
    img = table.get_pdf_image()
    if to_pdf_scale:
        ax.imshow(img, extent=(0, table.pdf_size[0], 0, table.pdf_size[1]))
    else:
        ax.imshow(img)


def draw_parse_constraints(table, ax):
    """Draw any user provided constraints (area, region, columns, etc)

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)

    ax : matplotlib.axes.Axes

    """
    if table.parse_details:
        # Display a bbox per region
        for region_str in table.parse_details["table_regions"] or []:
            draw_labeled_bbox(
                ax, bbox_from_str(region_str),
                "region: ({region_str})".format(region_str=region_str),
                color="purple",
                linestyle="dotted",
                linewidth=1,
                label_pos="bottom,right"
            )
        # Display a bbox per area
        for area_str in table.parse_details["table_areas"] or []:
            draw_labeled_bbox(
                ax, bbox_from_str(area_str),
                "area: ({area_str})".format(area_str=area_str),
                color="pink",
                linestyle="dotted",
                linewidth=1,
                label_pos="bottom,right"
            )


def prepare_plot(table, ax=None, to_pdf_scale=True):
    """Initialize plot and draw common components

    Parameters
    ----------
    table : camelot.core.Table
    ax : matplotlib.axes.Axes (optional)
    to_pdf_scale :

    ax : matplotlib.axes.Axes

    to_pdf_scale : bool (optional)

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")
    draw_pdf(table, ax, to_pdf_scale)
    draw_parse_constraints(table, ax)
    return ax


class PlotMethods():
    def __call__(self, table, kind="text", filename=None, ax=None):
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
        return plot_method(table, ax)

    @staticmethod
    def text(table, ax=None):
        """Generates a plot for all text elements present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
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
        return ax.get_figure()

    @staticmethod
    def grid(table, ax=None):
        """Generates a plot for the detected table grids
        on the PDF page.

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
        """Generates a plot for all table boundaries present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        _FOR_LATTICE = table.flavor == "lattice"
        ax = prepare_plot(table, ax, to_pdf_scale=not _FOR_LATTICE)

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
                        (t[0], t[1]), t[2] - t[0], t[3] - t[1],
                        color="blue",
                        alpha=0.5
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
        return ax.get_figure()

    @staticmethod
    def textedge(table, ax=None):
        """Generates a plot for relevant textedges.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax)
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(
                patches.Rectangle(
                    (t[0], t[1]), t[2] - t[0], t[3] - t[1],
                    color="blue",
                    alpha=0.2
                )
            )
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)

        if table.flavor == "hybrid":
            for network in table.parse_details["network_searches"]:
                most_connected_tl = network.most_connected_textline()

                ax.add_patch(
                    patches.Rectangle(
                        (most_connected_tl.x0, most_connected_tl.y0),
                        most_connected_tl.x1 - most_connected_tl.x0,
                        most_connected_tl.y1 - most_connected_tl.y0,
                        color="red",
                        alpha=0.5
                    )
                )
                for tl, alignments in network._textline_to_alignments.items():
                    coords = get_textline_coords(tl)
                    alignment_id_h, tls_h = alignments.max_v()
                    alignment_id_v, tls_v = alignments.max_h()
                    xs = list(map(lambda tl: tl.x0, tls_v))
                    ys = list(map(lambda tl: tl.y1, tls_h))
                    top_h = max(ys)
                    ax.text(
                        coords[alignment_id_h],
                        top_h + 5,
                        "{max_h_count}".format(max_h_count=len(tls_h)),
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=8,
                        color="green"
                    )
                    ax.plot(
                        [coords[alignment_id_h]] * len(ys), ys,
                        color="green",
                        linestyle="solid",
                        linewidth=1,
                        marker="o",
                        markersize=3
                    )

                    left_v = min(map(lambda tl: tl.x0, tls_v))
                    ax.text(
                        left_v - 5,
                        coords[alignment_id_v],
                        "{max_v_count}".format(max_v_count=len(tls_v)),
                        verticalalignment="center",
                        horizontalalignment="right",
                        fontsize=8,
                        color="blue"
                    )
                    ax.plot(
                        xs, [coords[alignment_id_v]] * len(xs),
                        color="blue",
                        linestyle="solid",
                        linewidth=1,
                        marker="o",
                        markersize=3
                    )
        else:
            for te in table._textedges:
                ax.plot([te.coord, te.coord], [te.y0, te.y1])
        return ax.get_figure()

    @staticmethod
    def joint(table, ax=None):
        """Generates a plot for all line intersections present
        on the PDF page.

        Parameters
        ----------
        table : camelot.core.Table
        ax : matplotlib.axes.Axes (optional)

        Returns
        -------
        fig : matplotlib.fig.Figure

        """
        ax = prepare_plot(table, ax, to_pdf_scale=False)
        table_bbox = table._bbox_unscaled
        x_coord = []
        y_coord = []
        for k in table_bbox.keys():
            for coord in table_bbox[k]:
                x_coord.append(coord[0])
                y_coord.append(coord[1])
        ax.plot(x_coord, y_coord, "ro")
        return ax.get_figure()

    @staticmethod
    def line(table, ax=None):
        """Generates a plot for all line segments present
        on the PDF page.

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
    def hybrid_table_search(table, ax=None):
        """Generates a plot illustrating the steps of the hybrid table search.

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
                    ax, bbox,
                    "box #{box_id} / iter #{iteration}".format(
                        box_id=box_id,
                        iteration=iteration
                    ),
                    color="red",
                    linewidth=5 if final else 2,
                    label_pos="bottom,left"
                )

                ax.add_patch(
                    patches.Rectangle(
                        (bbox[0]-max_h_gap, bbox[1]-max_v_gap),
                        bbox[2] - bbox[0] + 2 * max_h_gap,
                        bbox[3] - bbox[1] + 2 * max_v_gap,
                        color="orange",
                        fill=False
                    )
                )

        for box_id, col_search in enumerate(parse_details["col_searches"]):
            draw_labeled_bbox(
                ax, col_search["expanded_bbox"],
                "box body + header #{box_id}".format(
                    box_id=box_id
                ),
                color="red",
                linewidth=4,
                label_pos="top,left"
            )
            draw_labeled_bbox(
                ax, col_search["core_bbox"],
                "box body #{box_id}".format(
                    box_id=box_id
                ),
                color="orange",
                linewidth=2,
                label_pos="bottom,left"
            )

        return ax.get_figure()
