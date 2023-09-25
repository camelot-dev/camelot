try:
    import pypdfium2 as pdfium
except Exception as e:
    pdfium = None
    pdfium_exc = e
else:
    pdfium_exc = None


class PdfiumBackend:
    def convert(self, pdf_path, png_path, resolution=300):
        if not pdfium:
            raise OSError(f"pypdfium2 is not available: {pdfium_exc!r}")
        doc = pdfium.PdfDocument(pdf_path)
        assert len(doc) == 1
        doc.init_forms()
        image = doc[0].render(scale=resolution / 72).to_pil()
        image.save(png_path)
