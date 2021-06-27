# -*- coding: utf-8 -*-

from .poppler_backend import PopplerBackend
from .ghostscript_backend import GhostscriptBackend

backends = {"poppler": PopplerBackend, "ghostscript": GhostscriptBackend}


class ImageConversionBackend(object):
    def __init__(self, backend="poppler"):
        self.backend = backend

    def convert(self, pdf_path, png_path):
        converter = backends[self.backend]()
        converter.convert(pdf_path, png_path)
