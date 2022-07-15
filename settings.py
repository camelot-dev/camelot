import os

from betterconf import Config, field


class SetupConfig(Config):
    VERSION = (0, 10, 1)
    PRERELEASE = None  # alpha, beta or rc
    REVISION = None

    def generate_version(version, prerelease=None, revision=None):
        version_parts = [".".join(map(str, version))]
        if prerelease is not None:
            version_parts.append(f"-{prerelease}")
        if revision is not None:
            version_parts.append(f".{revision}")
        return "".join(version_parts)

    __title__ = field(default="camelot-py")
    __description__ = field(default="PDF Table Extraction for Humans.")
    __url__ = field(default="http://camelot-py.readthedocs.io/")
    __version__ = field(default=generate_version(VERSION, prerelease=PRERELEASE, revision=REVISION))
    __author__ = field(default="Vinayak Mehta")
    __author_email__ = field(default="vmehta94@gmail.com")
    __license__ = field(default="MIT License")


    with open("README.md", "r") as f:
        readme = f.read()

    requires = [
        "chardet>=3.0.4",
        "click>=6.7",
        "numpy>=1.13.3",
        "openpyxl>=2.5.8",
        "pandas>=0.23.4",
        "pdfminer.six>=20200726",
        "PyPDF2>=1.26.0",
        "tabulate>=0.8.9",
    ]

    base_requires = [
        "ghostscript>=0.7",
        "opencv-python>=3.4.2.17",
        "pdftopng>=0.2.3"
    ]

    plot_requires = [
        "matplotlib>=2.2.3",
    ]

    dev_requires = [
        "codecov>=2.0.15",
        "pytest>=5.4.3",
        "pytest-cov>=2.10.0",
        "pytest-mpl>=0.11",
        "pytest-runner>=5.2",
        "Sphinx>=3.1.2",
        "sphinx-autobuild>=2021.3.14",
    ]

    all_requires = base_requires + plot_requires
    dev_requires = field(default=dev_requires + all_requires)
