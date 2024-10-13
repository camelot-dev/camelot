import pytest

from camelot.backends import ImageConversionBackend


@pytest.fixture
def patch_backends(monkeypatch):
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS",
        {
            "poppler": PopplerBackendError,
            "ghostscript": GhostscriptBackendNoError,
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


def test_poppler_backend_error_when_no_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="poppler", use_fallback=False)

    message = "Image conversion failed with image conversion backend 'poppler'"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


def test_ghostscript_backend_when_use_fallback(patch_backends):
    backend = ImageConversionBackend(backend="ghostscript")
    backend.convert("foo", "bar")


def test_ghostscript_backend_error_when_use_fallback(monkeypatch):
    backends = {
        "ghostscript": GhostscriptBackendError,
        "poppler": PopplerBackendError,
    }

    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", backends, raising=True
    )
    backend = ImageConversionBackend(backend="ghostscript")

    message = "Image conversion failed with image conversion backend 'ghostscript'"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")
