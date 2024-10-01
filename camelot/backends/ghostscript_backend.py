

class GhostscriptBackend:

    def convert(self, pdf_path, png_path, resolution=300):
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
