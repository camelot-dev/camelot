"""Creates a ghostscript backend class to convert a pdf to a png file."""

from camelot.backends.base import ConversionBackend


class GhostscriptBackend(ConversionBackend):
    """Classmethod to create GhostscriptScriptBackend."""

    def convert(self, pdf_path: str, png_path: str, resolution: int = 300) -> None:
        """Convert a PDF to a PNG image using Ghostscript .

        Parameters
        ----------
        pdf_path : str
            [description]
        png_path : str
            [description]
        resolution : int, optional
            [description], by default 300

        Raises
        ------
        OSError
            [description]
        """
        try:
            import ghostscript  # type: ignore[import-untyped]
        except ModuleNotFoundError as ex:
            raise OSError(
                "Ghostscript is not installed. You can install it using the instructions"
                " here: https://camelot-py.readthedocs.io/en/latest/user/install-deps.html"
            ) from ex

        gs_command = [
            "gs",
            "-q",
            "-sDEVICE=png16m",
            "-o",
            png_path,
            f"-r{resolution}",
            pdf_path,
        ]
        ghostscript.Ghostscript(*gs_command)
