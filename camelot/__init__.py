import importlib.metadata
import logging
from typing import Optional

from .io import read_pdf
from .plotting import PlotMethods


def get_version() -> Optional[str]:
    """Retrieve the version number from package metadata."""
    try:
        return importlib.metadata.version("camelot-py")
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development environments
        return "0.0.0+unknown"


__version__ = get_version()


# set up logging
logger = logging.getLogger("camelot")

format_string = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(format_string, datefmt="%Y-%m-%dT%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)

# instantiate plot method
plot = PlotMethods()
