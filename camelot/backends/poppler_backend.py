# -*- coding: utf-8 -*-

import shutil
import subprocess


class PopplerBackend(object):
    def convert(self, pdf_path, png_path):
        from pdftopng.pdftopng import convert
        convert(pdf_path, png_path)
