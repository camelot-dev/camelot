# -*- coding: utf-8 -*-

import os
from setuptools import find_packages
from settings import SetupConfig
cfg = SetupConfig


def setup_package():
    metadata = dict(
        name=cfg.__title__,
        version=cfg.__version__,
        description=cfg.__description__,
        long_description=cfg.readme,
        long_description_content_type="text/markdown",
        url=cfg.__url__,
        author=cfg.__author__,
        author_email=cfg.__author_email__,
        license=cfg.__license__,
        packages=find_packages(exclude=("tests",)),
        install_requires=cfg.requires,
        extras_require={
            "all": cfg.all_requires,
            "base": cfg.base_requires,
            "cv": cfg.base_requires,  # deprecate
            "dev": cfg.dev_requires,
            "plot": cfg.plot_requires,
        },
        entry_points={
            "console_scripts": [
                "camelot = camelot.cli:cli",
            ],
        },
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
    )

    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
