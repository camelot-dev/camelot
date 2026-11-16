from unittest.mock import Mock

import pytest

from camelot.backends import ImageConversionBackend
from tests.doubles import GhostscriptBackendError
from tests.doubles import GhostscriptBackendNoError
from tests.doubles import PdfiumBackendError
from tests.doubles import PopplerBackendError


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


def test_poppler_backend_error_when_no_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="poppler", use_fallback=False)

    message = r"Image conversion failed with image conversion backend.+Poppler"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


def test_ghostscript_backend_when_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="ghostscript")
    backend.convert("foo", "bar")


def test_last_fallback_error_when_all_backends_fail(monkeypatch):
    """Report the last backend error after all conversion fallbacks fail."""
    backends = {
        "pdfium": PdfiumBackendError,
        "ghostscript": GhostscriptBackendError,
        "poppler": PopplerBackendError,
    }

    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", backends, raising=True
    )
    backend = ImageConversionBackend(backend="pdfium")

    message = "Image conversion failed with image conversion backend 'poppler'\n error: Image conversion failed"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


@pytest.mark.xfail
def test_invalid_backend():
    ImageConversionBackend(backend="invalid_backend")


def test_conversion_tries_next_fallback_after_failure(monkeypatch):
    successful_backend = Mock()
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS",
        {
            "pdfium": PdfiumBackendError,
            "ghostscript": GhostscriptBackendError,
            "poppler": lambda: successful_backend,
        },
    )

    ImageConversionBackend(backend="pdfium").convert("foo", "bar", page=3)

    successful_backend.convert.assert_called_once_with("foo", "bar", page=3)


def test_custom_backend_failure_is_not_swallowed():
    backend = ImageConversionBackend(backend=PdfiumBackendError())

    with pytest.raises(ValueError, match="Image conversion failed"):
        backend.convert("foo", "bar")
