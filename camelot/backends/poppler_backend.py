import shutil
import subprocess


class PopplerBackend:
    def convert(self, pdf_path, png_path):
        pdftopng_executable = shutil.which("pdftopng")
        if pdftopng_executable is None:
            raise OSError(
                "pdftopng is not installed. You can install it using the 'pip install pdftopng' command."
            )

        pdftopng_command = [pdftopng_executable, pdf_path, png_path]

        try:
            subprocess.check_output(
                " ".join(pdftopng_command), stderr=subprocess.STDOUT, shell=True
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output)
