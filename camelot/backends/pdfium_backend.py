"""Creates a pdfium backend class to convert a pdf to a png file."""

from camelot.backends.base import ConversionBackend


try:
    import pypdfium2 as pdfium
except Exception as e:
    pdfium = None
    pdfium_exc = e
else:
    pdfium_exc = None


class PdfiumBackend(ConversionBackend):
    """Classmethod to create PdfiumBackend."""

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
            Raise an error if pdfium is not installed
        """
        if not pdfium:
            raise OSError(f"pypdfium2 is not available: {pdfium_exc!r}")
        doc = pdfium.PdfDocument(pdf_path)
        doc.init_forms()
        image = doc[0].render(scale=resolution / 72).to_pil()
        image.save(png_path)
