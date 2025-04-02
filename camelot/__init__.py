import logging
import warnings

def _issue_deprecation_warning():
    """Issues a deprecation warning about the library's status."""
    message = (
        "pypdf_table_extraction is deprecated and no longer actively maintained. "
        "Please use camelot-py instead: https://github.com/camelot-dev/camelot"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)

# Issue the warning when the module is imported
_issue_deprecation_warning()

from .__version__ import __version__  # noqa D100, F400
from .io import read_pdf
from .plotting import PlotMethods

# set up logging
logger = logging.getLogger("camelot")

format_string = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(format_string, datefmt="%Y-%m-%dT%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)

# instantiate plot method
plot = PlotMethods()
