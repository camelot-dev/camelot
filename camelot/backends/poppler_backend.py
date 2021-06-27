# -*- coding: utf-8 -*-

from pdftopng import pdftopng


class PopplerBackend(object):
    def convert(self, pdf_path, png_path):
        pdftopng.convert(pdf_path, png_path)
