"""Regression tests for the flavor='auto' detection probe (_detect_flavor)."""

import camelot
from camelot.io import _detect_flavor


def test_detect_flavor_returns_lattice_for_ruled_pdf(foo_pdf):
    # foo.pdf is a ruled (lattice) table. The probe must classify it as
    # 'lattice'. Regression: _detect_flavor passed a non-existent
    # `resolution=` kwarg to backend.convert(), which raised TypeError that
    # the bare except swallowed, so 'auto' silently fell back to 'network'
    # for *every* PDF.
    assert _detect_flavor(foo_pdf) == "lattice"


def test_auto_flavor_extracts_ruled_table(foo_pdf):
    # End-to-end: flavor='auto' on a ruled PDF should pick lattice and
    # extract the table (not mis-route to network).
    tables = camelot.read_pdf(foo_pdf, flavor="auto")
    assert len(tables) == 1
    assert tables[0].shape[0] >= 2 and tables[0].shape[1] >= 2


def test_auto_routes_lattice_pages_through_combined_engine(foo_pdf, monkeypatch):
    # (#763) auto must parse ruled pages with lattice + engine='combined',
    # the strongest detector.
    import camelot.handlers as handlers_mod

    seen = {}
    original = handlers_mod.PDFHandler.parse

    def spy(self, *args, **kwargs):
        if kwargs.get("flavor") == "lattice":
            seen["engine"] = kwargs.get("engine")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(handlers_mod.PDFHandler, "parse", spy)
    camelot.read_pdf(foo_pdf, flavor="auto")
    assert seen.get("engine") == "combined"


def test_auto_render_cache_avoids_double_render(foo_pdf, monkeypatch):
    # The page rendered for the auto probe is reused by the lattice parse,
    # so a ruled page is rasterised exactly once, not twice.
    import camelot.backends.image_conversion as ic

    rendered_pages = []
    original = ic.ImageConversionBackend.convert

    def spy(self, pdf_path, png_path, page=1):
        rendered_pages.append(page)
        return original(self, pdf_path, png_path, page=page)

    monkeypatch.setattr(ic.ImageConversionBackend, "convert", spy)
    camelot.read_pdf(foo_pdf, flavor="auto")  # foo.pdf: 1 ruled page
    assert rendered_pages.count(1) == 1
