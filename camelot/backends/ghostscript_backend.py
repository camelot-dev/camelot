"""Creates a ghostscript backend class to convert a pdf to a png file."""


class GhostscriptBackend:
    """Classmethod to create GhostscriptScriptBackend."""

    def convert(self, pdf_path, png_path, resolution=300):
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
            import ghostscript
        except RuntimeError:
            raise OSError(
                "Ghostscript is not installed. You can install it using the instructions"
                " here: https://pypdf-table-extraction.readthedocs.io/en/latest/user/install-deps.html"
            )

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
