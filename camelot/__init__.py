import logging

from .__version__ import __version__
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
