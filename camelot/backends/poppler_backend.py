"""Creates a poppler backend class to convert a pdf to a png file.

Raises
------
OSError
    [description]
ValueError
    [description]
"""

import os
import shutil
import subprocess  # noqa
import sys

from camelot.backends.base import ConversionBackend


path = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]


class PopplerBackend(ConversionBackend):
    """Classmethod to create a poplerBackendBackend class."""

    def convert(self, pdf_path: str, png_path: str, resolution: int = 300, page: int = 1) -> None:
        """Convert PDF to png.

        Parameters
        ----------
        pdf_path : str
            Path where to read the pdf file.
        png_path : str
            Path where to save png file.
        page: int, optional
            Single page to convert.

        Raises
        ------
        OSError
            [description]
        ValueError
            [description]
        """
        pdftopng_executable = shutil.which("pdftocairo", path=path)
        if pdftopng_executable is None:
            raise OSError(
                "pdftopng is not installed. You can install it using the 'pip install pdftopng' command."
            )

        png_stem, _ = os.path.splitext(png_path)
        pdftopng_command = [pdftopng_executable, "-png", "-singlefile",
                            "-f", str(page), "-l", str(page), pdf_path, png_stem]

        try:
            subprocess.check_output(
                " ".join(pdftopng_command),
                stderr=subprocess.STDOUT,
                shell=False,  # noqa
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output) from e
