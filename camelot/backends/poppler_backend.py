# -*- coding: utf-8 -*-

import subprocess


class PopplerBackend(object):
    def convert(self, pdf_path, png_path):
        pdftopng_command = ["pdftopng", pdf_path, png_path]

        try:
            subprocess.check_output(
                " ".join(pdftopng_command), stderr=subprocess.STDOUT, shell=True
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output)
