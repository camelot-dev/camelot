import shutil
import subprocess


class PopplerBackend:
    def convert(self, pdf_path, png_path):
        from pdftopng.pdftopng import convert
        convert(pdf_path, png_path)
