"""Test doubles shared by the test modules.

Kept apart from the test files so each test module sticks to one style
(plain test functions) instead of mixing module-level classes in.
"""


class PopplerBackendError:
    def convert(self, pdf_path, png_path, page=1):
        raise ValueError("Image conversion failed")


class GhostscriptBackendError:
    def convert(self, pdf_path, png_path, page=1):
        raise ValueError("Image conversion failed")


class GhostscriptBackendNoError:
    def convert(self, pdf_path, png_path, page=1):
        pass


class PdfiumBackendError:
    def convert(self, pdf_path, png_path, page=1):
        raise ValueError("Image conversion failed")


class TextlineStub:
    """Minimal stand-in for a PDFMiner LTTextLine sufficient for get_table_index."""

    __slots__ = ("x0", "y0", "x1", "y1", "_objs", "_text")

    def __init__(self, x0, y0, x1, y1, text="x"):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self._objs = []
        self._text = text

    def get_text(self):
        return self._text + "\n"
