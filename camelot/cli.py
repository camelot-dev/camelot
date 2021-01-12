# -*- coding: utf-8 -*-

import logging

import click

try:
    import matplotlib.pyplot as plt
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True

from . import __version__, read_pdf, plot


logger = logging.getLogger("camelot")
logger.setLevel(logging.INFO)


class Config(object):
    def __init__(self):
        self.config = {}

    def set_config(self, key, value):
        self.config[key] = value


pass_config = click.make_pass_decorator(Config)


@click.group(name="camelot")
@click.version_option(version=__version__)
@click.option("-q", "--quiet", is_flag=False, help="Suppress logs and warnings.")
@click.option(
    "-p",
    "--pages",
    default="1",
    help="Comma-separated page numbers." " Example: 1,3,4 or 1,4-end or all.",
)
@click.option("-pw", "--password", help="Password for decryption.")
@click.option("-o", "--output", help="Output file path.")
@click.option(
    "-f",
    "--format",
    type=click.Choice(["csv", "excel", "html", "json", "markdown", "sqlite"]),
    help="Output file format.",
)
@click.option("-z", "--zip", is_flag=True, help="Create ZIP archive.")
@click.option(
    "-split",
    "--split_text",
    is_flag=True,
    help="Split text that spans across multiple cells.",
)
@click.option(
    "-flag",
    "--flag_size",
    is_flag=True,
    help="Flag text based on" " font size. Useful to detect super/subscripts.",
)
@click.option(
    "-strip",
    "--strip_text",
    help="Characters that should be stripped from a string before"
    " assigning it to a cell.",
)
@click.option(
    "-M",
    "--margins",
    nargs=3,
    default=(1.0, 0.5, 0.1),
    help="PDFMiner char_margin, line_margin and word_margin.",
)
@click.pass_context
def cli(ctx, *args, **kwargs):
    """Camelot: PDF Table Extraction for Humans"""
    ctx.obj = Config()
    for key, value in kwargs.items():
        ctx.obj.set_config(key, value)


@cli.command("lattice")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-back", "--process_background", is_flag=True, help="Process background lines."
)
@click.option(
    "-scale",
    "--line_scale",
    default=15,
    help="Line size scaling factor. The larger the value,"
    " the smaller the detected lines.",
)
@click.option(
    "-copy",
    "--copy_text",
    default=[],
    type=click.Choice(["h", "v"]),
    multiple=True,
    help="Direction in which text in a spanning cell" " will be copied over.",
)
@click.option(
    "-shift",
    "--shift_text",
    default=["l", "t"],
    type=click.Choice(["", "l", "r", "t", "b"]),
    multiple=True,
    help="Direction in which text in a spanning cell will flow.",
)
@click.option(
    "-l",
    "--line_tol",
    default=2,
    help="Tolerance parameter used to merge close vertical" " and horizontal lines.",
)
@click.option(
    "-j",
    "--joint_tol",
    default=2,
    help="Tolerance parameter used to decide whether"
    " the detected lines and points lie close to each other.",
)
@click.option(
    "-block",
    "--threshold_blocksize",
    default=15,
    help="For adaptive thresholding, size of a pixel"
    " neighborhood that is used to calculate a threshold value for"
    " the pixel. Example: 3, 5, 7, and so on.",
)
@click.option(
    "-const",
    "--threshold_constant",
    default=-2,
    help="For adaptive thresholding, constant subtracted"
    " from the mean or weighted mean. Normally, it is positive but"
    " may be zero or negative as well.",
)
@click.option(
    "-I",
    "--iterations",
    default=0,
    help="Number of times for erosion/dilation will be applied.",
)
@click.option(
    "-res",
    "--resolution",
    default=300,
    help="Resolution used for PDF to PNG conversion.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "joint", "line"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def lattice(c, *args, **kwargs):
    """Use lines between text to parse the table."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    copy_text = list(kwargs["copy_text"])
    kwargs["copy_text"] = None if not copy_text else copy_text
    kwargs["shift_text"] = list(kwargs["shift_text"])

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath, pages=pages, flavor="lattice", suppress_stdout=quiet, **kwargs
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)


@cli.command("stream")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-C",
    "--columns",
    default=[],
    multiple=True,
    help="X coordinates of column separators.",
)
@click.option(
    "-e",
    "--edge_tol",
    default=50,
    help="Tolerance parameter" " for extending textedges vertically.",
)
@click.option(
    "-r",
    "--row_tol",
    default=2,
    help="Tolerance parameter" " used to combine text vertically, to generate rows.",
)
@click.option(
    "-c",
    "--column_tol",
    default=0,
    help="Tolerance parameter"
    " used to combine text horizontally, to generate columns.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "textedge"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def stream(c, *args, **kwargs):
    """Use spaces between text to parse the table."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    columns = list(kwargs["columns"])
    kwargs["columns"] = None if not columns else columns

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath, pages=pages, flavor="stream", suppress_stdout=quiet, **kwargs
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)
