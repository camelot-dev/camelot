# -*- coding: utf-8 -*-

import ghostscript


class GhostscriptBackend(object):
    def convert(self, pdf_path, png_path, resolution=300):
        gs_args = [
            "gs",
            "-q",
            "-sDEVICE=png16m",
            "-o",
            png_path,
            f"-r{resolution}",
            pdf_path,
        ]
        ghostscript.Ghostscript(*gs_args)
