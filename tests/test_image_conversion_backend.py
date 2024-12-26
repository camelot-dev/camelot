import pytest

from camelot.backends import ImageConversionBackend


@pytest.fixture
def patch_backends(monkeypatch):
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS",
        {
            "poppler": PopplerBackendError,
            "ghostscript": GhostscriptBackendNoError,
            "pdfdium": PdfiumBackendError,
        },
        raising=True,
    )


class PopplerBackendError:
    def convert(self, pdf_path, png_path):
        raise ValueError("Image conversion failed")


class GhostscriptBackendError:
    def convert(self, pdf_path, png_path):
        raise ValueError("Image conversion failed")


class GhostscriptBackendNoError:
    def convert(self, pdf_path, png_path):
        pass


class PdfiumBackendError:
    def convert(self, pdf_path, png_path):
        raise ValueError("Image conversion failed")


def test_poppler_backend_error_when_no_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="poppler", use_fallback=False)

    message = r"Image conversion failed with image conversion backend.+Poppler"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


def test_ghostscript_backend_when_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="ghostscript")
    backend.convert("foo", "bar")


def test_ghostscript_backend_error_when_use_fallback(monkeypatch):
    """Use an image conversion backend and let it fallback to ghostscript.

    Then capture the error message of the second backend (the fallback).
    """
    backends = {
        "pdfium": PdfiumBackendError,
        "ghostscript": GhostscriptBackendError,
        "poppler": PopplerBackendError,
    }

    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", backends, raising=True
    )
    backend = ImageConversionBackend(backend="pdfium")

    message = "Image conversion failed with image conversion backend 'ghostscript'\n error: Image conversion failed"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


@pytest.mark.xfail
def test_invalid_backend():
    ImageConversionBackend(backend="invalid_backend")
