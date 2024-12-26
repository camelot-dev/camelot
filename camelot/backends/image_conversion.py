"""Classes and functions for the ImageConversionBackend backends."""

from typing import Any
from typing import Dict
from typing import List
from typing import Type

from .base import ConversionBackend
from .ghostscript_backend import GhostscriptBackend
from .pdfium_backend import PdfiumBackend
from .poppler_backend import PopplerBackend


BACKENDS: Dict[str, Type[ConversionBackend]] = {
    "pdfium": PdfiumBackend,
    "ghostscript": GhostscriptBackend,
    "poppler": PopplerBackend,
}


class ImageConversionError(ValueError):  # noqa D101
    pass


class ImageConversionBackend:
    """Classes the ImageConversionBackend backend."""

    def __init__(self, backend: Any = "pdfium", use_fallback: bool = True) -> None:
        """Initialize the conversion backend .

        Parameters
        ----------
        backend : Any, optional
            Backend for image conversion, by default "pdfium"
        use_fallback : bool, optional
            Fallback to another backend if unavailable, by default True

        Raises
        ------
        ValueError
            Raise an error if the backend is not supported.
        """
        self.backend: ConversionBackend = self.get_backend(backend)
        self.use_fallback: bool = use_fallback
        self.fallbacks: List[str] = list(
            filter(lambda x: isinstance(backend, str) and x != backend, BACKENDS.keys())
        )

    def get_backend(self, backend):
        """Retrieve the specified backend for processing.

        This method checks if the provided backend is a string representing
        a known backend or an object implementing a 'convert' method. It
        returns an instance of the backend if valid.

        Parameters
        ----------
        backend : str or object
            The backend to retrieve. This can be:
                - A string ('poppler' or 'ghostscript') corresponding to a pre-defined backend.
                - An object that must implement a 'convert' method.

        Returns
        -------
        object
            An instance of the specified backend.

        Raises
        ------
        NotImplementedError
            If the backend is a string that is not recognized or if it is an
            object that does not implement the 'convert' method.

        Examples
        --------
        >> backend_instance = get_backend('poppler')
        >> backend_instance = get_backend(my_custom_backend)

        Notes
        -----
        The valid backends are defined in the BACKENDS dictionary. The
        method verifies the backend type and raises exceptions for
        unsupported backends.
        """

        def implements_convert():
            methods = [
                method for method in dir(backend) if method.startswith("__") is False
            ]
            return "convert" in methods

        if isinstance(backend, str):
            if backend not in BACKENDS.keys():
                raise NotImplementedError(
                    f"Unknown backend {backend!r} specified. Please use 'pdfium', 'poppler' or 'ghostscript'."
                )

            return BACKENDS[backend]()
        else:
            if not implements_convert():
                raise NotImplementedError(
                    f"{backend!r} must implement a 'convert' method"
                )

            return backend

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
            self.backend.convert(pdf_path, png_path)
        except Exception as f:
            if self.use_fallback:
                for fallback in self.fallbacks:
                    try:
                        converter = BACKENDS[fallback]()
                        converter.convert(pdf_path, png_path)
                    except Exception as e:
                        msg = f"Image conversion failed with image conversion backend {fallback!r}\n error: {e}"
                        raise ImageConversionError(msg) from e
                    else:
                        break
            else:
                msg = f"Image conversion failed with image conversion backend {self.backend!r}\n error: {f}"
                raise ImageConversionError(msg) from f
