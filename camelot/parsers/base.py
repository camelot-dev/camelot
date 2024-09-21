import os

from ..core import Table
from ..utils import get_page_layout
from ..utils import get_text_objects


class BaseParser:
    """Defines a base parser."""

    def __init__(self, parser_id):
        self.id = parser_id

        # For plotting details of parsing algorithms
        self.debug_info = {}

    def _generate_layout(self, filename, layout, dimensions, page_idx, layout_kwargs):
        self.filename = filename
        self.layout_kwargs = layout_kwargs
        self.layout = layout
        self.dimensions = dimensions
        self.page = page_idx
        self.images = get_text_objects(self.layout, ltype="image")
        self.horizontal_text = get_text_objects(self.layout, ltype="horizontal_text")
        self.vertical_text = get_text_objects(self.layout, ltype="vertical_text")
        self.pdf_width, self.pdf_height = self.dimensions
        self.rootname, __ = os.path.splitext(self.filename)
        self.imagename = "".join([self.rootname, ".png"])

    """Initialize new table object, ready to be populated

    Parameters
    ----------
    table_idx : int
        Index of this table within the pdf page analyzed
    cols : list
        list of coordinate boundaries tuples (left, right)
    rows : list
        list of coordinate boundaries tuples (bottom, top)

    Returns
    -------
    table : camelot.core.Table

    """

    def _initialize_new_table(self, table_idx, cols, rows):
        table = Table(cols, rows)
        table.page = self.page
        table.order = table_idx + 1
        return table
