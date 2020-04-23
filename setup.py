# -*- coding: utf-8 -*-
from setuptools import setup
import os


here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'camelot', '__version__.py'), 'r') as f:
    exec(f.read(), about)

__version__ = about["__version__"]

cv_requires = [
    'opencv-python>=3.4.2.17'
]

plot_requires = [
    'matplotlib>=2.2.3',
]

dev_requires = [
    'codecov>=2.0.15',
    'pytest>=3.8.0',
    'pytest-cov>=2.6.0',
    'pytest-mpl>=0.10',
    'pytest-runner>=4.2',
    'Sphinx>=1.7.9'
]

all_requires = cv_requires + plot_requires
dev_requires = dev_requires + all_requires

if __name__ == "__main__":
    setup(
        extras_require={
            'all': all_requires,
            'cv': cv_requires,
            'dev': dev_requires,
            'plot': plot_requires
        },
        use_scm_version=True
    )
