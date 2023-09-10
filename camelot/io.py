import warnings
from pathlib import Path
from typing import Union

from pypdf._utils import StrByteType

from .handlers import PDFHandler
from .utils import remove_extra
from .utils import validate_input


def read_pdf(
    filepath: Union[StrByteType, Path],
    pages="1",
    password=None,
    flavor="lattice",
    suppress_stdout=False,
    layout_kwargs=None,
    **kwargs
):
    """Read PDF and return extracted tables.

    Note: kwargs annotated with ^ can only be used with flavor='stream'
    and kwargs annotated with * can only be used with flavor='lattice'.

    Parameters
    ----------
    filepath : str, Path, IO
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    flavor : str (default: 'lattice')
        The parsing method to use ('lattice' or 'stream').
        Lattice is used by default.
    suppress_stdout : bool, optional (default: True)
        Print all logs and warnings.
    layout_kwargs : dict, optional (default: {})
        A dict of `pdfminer.layout.LAParams
        <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
    table_areas : list, optional (default: None)
        List of table area strings of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    columns^ : list, optional (default: None)
        List of column x-coordinates strings where the coordinates
        are comma-separated.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    row_tol^ : int, optional (default: 2)
        Tolerance parameter used to combine text vertically,
        to generate rows.
    column_tol^ : int, optional (default: 0)
        Tolerance parameter used to combine text horizontally,
        to generate columns.
    process_background* : bool, optional (default: False)
        Process background lines.
    line_scale* : int, optional (default: 15)
        Line size scaling factor. The larger the value the smaller
        the detected lines. Making it very large will lead to text
        being detected as lines.
    copy_text* : list, optional (default: None)
        {'h', 'v'}
        Direction in which text in a spanning cell will be copied
        over.
    shift_text* : list, optional (default: ['l', 't'])
        {'l', 'r', 't', 'b'}
        Direction in which text in a spanning cell will flow.
    line_tol* : int, optional (default: 2)
        Tolerance parameter used to merge close vertical and horizontal
        lines.
    joint_tol* : int, optional (default: 2)
        Tolerance parameter used to decide whether the detected lines
        and points lie close to each other.
    threshold_blocksize* : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    threshold_constant* : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    iterations* : int, optional (default: 0)
        Number of times for erosion/dilation is applied.

        For more information, refer `OpenCV's dilate
        <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.
    resolution* : int, optional (default: 300)
        Resolution used for PDF to PNG conversion.

    Returns
    -------
    tables : camelot.core.TableList

    """
    if layout_kwargs is None:
        layout_kwargs = {}
    if flavor not in ["lattice", "stream"]:
        raise NotImplementedError(
            "Unknown flavor specified." " Use either 'lattice' or 'stream'"
        )

    with warnings.catch_warnings():
        if suppress_stdout:
            warnings.simplefilter("ignore")

        validate_input(kwargs, flavor=flavor)
        p = PDFHandler(filepath, pages=pages, password=password)
        kwargs = remove_extra(kwargs, flavor=flavor)
        tables = p.parse(
            flavor=flavor,
            suppress_stdout=suppress_stdout,
            layout_kwargs=layout_kwargs,
            **kwargs
        )
        return tables
