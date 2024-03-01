"""Classes and functions for the ImageConversionBackend backends."""

from typing import Dict
from typing import List
from typing import Type

from .base import ConversionBackend
from .ghostscript_backend import GhostscriptBackend
from .poppler_backend import PopplerBackend


BACKENDS: Dict[str, Type[ConversionBackend]] = {
    "poppler": PopplerBackend,
    "ghostscript": GhostscriptBackend,
}


class ImageConversionError(ValueError):  # noqa D101
    pass


class ImageConversionBackend:
    """Classes the ImageConversionBackend backend."""

    def __init__(self, backend: str = "poppler", use_fallback: bool = True) -> None:
        """Initialize the conversion backend .

        Parameters
        ----------
        backend : str, optional
            Backend for image conversion, by default "poppler"
        use_fallback : bool, optional
            Fallback to another backend if unavailable, by default True

        Raises
        ------
        ValueError
            Raise an error if the backend is not supported.
        """
        if backend not in BACKENDS.keys():
            raise ValueError(f"Image conversion backend {backend!r} not supported")

        self.backend: str = backend
        self.use_fallback: bool = use_fallback
        self.fallbacks: List[str] = list(
            filter(lambda x: x != backend, BACKENDS.keys())
        )

    def convert(self, pdf_path: str, png_path: str) -> None:
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
        except Exception as f:
            if self.use_fallback:
                for fallback in self.fallbacks:
                    try:
                        converter = BACKENDS[fallback]()
                        converter.convert(pdf_path, png_path)
                    except Exception as e:
                        msg = f"Image conversion failed with image conversion backend {fallback!r}"
                        raise ImageConversionError(msg) from e
                    else:
                        break
            else:
                msg = f"Image conversion failed with image conversion backend {self.backend!r}"
                raise ImageConversionError(msg) from f
