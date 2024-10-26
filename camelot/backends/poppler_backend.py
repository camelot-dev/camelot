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

    def convert(self, pdf_path: str, png_path: str, resolution: int = 300) -> None:
        """Convert PDF to png.

        Parameters
        ----------
        pdf_path : str
            Path where to read the pdf file.
        png_path : str
            Path where to save png file.

        Raises
        ------
        OSError
            [description]
        ValueError
            [description]
        """
        pdftopng_executable = shutil.which("pdftopng", path=path)
        if pdftopng_executable is None:
            raise OSError(
                "pdftopng is not installed. You can install it using the 'pip install pdftopng' command."
            )

        pdftopng_command = [pdftopng_executable, pdf_path, png_path]

        try:
            subprocess.check_output(
                " ".join(pdftopng_command),
                stderr=subprocess.STDOUT,
                shell=False,  # noqa
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output) from e
