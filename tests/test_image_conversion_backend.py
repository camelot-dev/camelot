# -*- coding: utf-8 -*-

import pytest

import camelot.backends.image_conversion
from camelot.backends import ImageConversionBackend


class PopplerBackendError(object):
    def convert(self, pdf_path, png_path):
        raise ValueError("Image conversion failed")


class GhostscriptBackendError(object):
    def convert(self, pdf_path, png_path):
        raise ValueError("Image conversion failed")


class GhostscriptBackendNoError(object):
    def convert(self, pdf_path, png_path):
        pass


def test_poppler_backend_error_when_no_use_fallback(monkeypatch):
    BACKENDS = {
        "poppler": PopplerBackendError,
        "ghostscript": GhostscriptBackendNoError,
    }
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", BACKENDS, raising=True
    )
    backend = ImageConversionBackend(use_fallback=False)

    message = "Image conversion failed with image conversion backend 'poppler'"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")


def test_ghostscript_backend_when_use_fallback(monkeypatch):
    BACKENDS = {
        "poppler": PopplerBackendError,
        "ghostscript": GhostscriptBackendNoError,
    }
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", BACKENDS, raising=True
    )
    backend = ImageConversionBackend()
    backend.convert("foo", "bar")


def test_ghostscript_backend_error_when_use_fallback(monkeypatch):
    BACKENDS = {"poppler": PopplerBackendError, "ghostscript": GhostscriptBackendError}
    monkeypatch.setattr(
        "camelot.backends.image_conversion.BACKENDS", BACKENDS, raising=True
    )
    backend = ImageConversionBackend()

    message = "Image conversion failed with image conversion backend 'ghostscript'"
    with pytest.raises(ValueError, match=message):
        backend.convert("foo", "bar")
