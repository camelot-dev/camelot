from .pdfium_backend import PdfiumBackend
from .ghostscript_backend import GhostscriptBackend
from .poppler_backend import PopplerBackend


BACKENDS = {
    "pdfium": PdfiumBackend,
    "poppler": PopplerBackend,
    "ghostscript": GhostscriptBackend,
}


class ImageConversionBackend:
    def __init__(self, backend="pdfium", use_fallback=True):
        if backend not in BACKENDS.keys():
            raise ValueError(f"Image conversion backend '{backend}' not supported")

        self.backend = backend
        self.use_fallback = use_fallback
        self.fallbacks = list(BACKENDS.keys())
        self.fallbacks.remove(self.backend)

    def convert(self, pdf_path, png_path):
        try:
            converter = BACKENDS[self.backend]()
            converter.convert(pdf_path, png_path)
        except Exception as e:
            import sys

            if self.use_fallback:
                for fallback in self.fallbacks:
                    try:
                        converter = BACKENDS[fallback]()
                        converter.convert(pdf_path, png_path)
                    except Exception as e:
                        raise type(e)(
                            str(e) + f" with image conversion backend '{fallback}'"
                        ).with_traceback(sys.exc_info()[2])
                        continue
                    else:
                        break
            else:
                raise type(e)(
                    str(e) + f" with image conversion backend '{self.backend}'"
                ).with_traceback(sys.exc_info()[2])
