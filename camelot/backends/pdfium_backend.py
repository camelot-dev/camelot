"""Creates a pdfium backend class to convert a pdf to a png file."""

from camelot.backends.base import ConversionBackend

PDFIUM_EXC = None

try:
    import pypdfium2 as pdfium

except ModuleNotFoundError as e:
    PDFIUM_EXC = e


class PdfiumBackend(ConversionBackend):
    """Classmethod to create PdfiumBackend."""

    def installed(self) -> bool:  # noqa D102
        if not PDFIUM_EXC:
            return True
        return False

    def convert(
        self, pdf_path: str, png_path: str, resolution: int = 300, page: int = 1
    ) -> None:
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
            Raise an error if pdfium is not installed
        """
        if not self.installed():
            raise OSError(f"pypdfium2 is not available: {PDFIUM_EXC!r}")
        doc = pdfium.PdfDocument(pdf_path)
        try:
            doc.init_forms()
            image = doc[page - 1].render(scale=resolution / 72).to_pil()
            try:
                image.save(png_path)
            finally:
                image.close()
        finally:
            doc.close()

    def to_array(self, pdf_path: str, resolution: int = 300, page: int = 1):
        """Render a page straight to a BGR uint8 ndarray — no PNG round-trip.

        Same pixels as :meth:`convert` would have written, returned in
        memory in OpenCV's BGR channel order (so it's a drop-in for
        ``cv2.imread`` of that PNG). Skips the PNG encode+decode, which is
        ~a quarter of the lattice raster time.
        """
        import numpy as np

        if not self.installed():
            raise OSError(f"pypdfium2 is not available: {PDFIUM_EXC!r}")
        doc = pdfium.PdfDocument(pdf_path)
        try:
            doc.init_forms()
            image = doc[page - 1].render(scale=resolution / 72).to_pil()
            try:
                rgb = np.asarray(image.convert("RGB"))
                return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR
            finally:
                image.close()
        finally:
            doc.close()
