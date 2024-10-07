"""Classes tand functions for the ImageConversionBackend backends."""

from .ghostscript_backend import GhostscriptBackend
from .poppler_backend import PopplerBackend


BACKENDS = {"poppler": PopplerBackend, "ghostscript": GhostscriptBackend}


class ImageConversionBackend:
    """Classes the ImageConversionBackend backend."""

    def __init__(self, backend="poppler", use_fallback=True):
        """Initialize the conversion backend .

        Parameters
        ----------
        backend : str, optional
            [description], by default "poppler"
        use_fallback : bool, optional
            [description], by default True

        Raises
        ------
        ValueError
            [description]
        """
        if backend not in BACKENDS.keys():
            raise ValueError(f"Image conversion backend {backend!r} not supported")

        self.backend = backend
        self.use_fallback = use_fallback
        self.fallbacks = list(filter(lambda x: x != backend, BACKENDS.keys()))

    def convert(self, pdf_path, png_path):
        """Convert PDF to png_path.

        Parameters
        ----------
        pdf_path : str
            Path where to read the pdf file.
        png_path : str
            Path where to save png file.

        Raises
        ------
        type
            [description]
        type
            [description]
        """
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
                            str(e) + f" with image conversion backend {fallback!r}"
                        ).with_traceback(sys.exc_info()[2])
                        continue
                    else:
                        break
            else:
                raise type(e)(
                    str(e) + f" with image conversion backend {self.backend!r}"
                ).with_traceback(sys.exc_info()[2])
