try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None


class PdfiumBackend:
    def convert(self, pdf_path, png_path, resolution=300):
        if not pdfium:
            raise OSError("pypdfium2 is not installed.")
        doc = pdfium.PdfDocument(pdf_path)
        assert len(doc) == 1
        doc.init_forms()
        image = doc[0].render(scale=resolution / 72).to_pil()
        image.save(png_path)
