# -*- coding: utf-8 -*-

import pytest

from camelot.__version__ import  generate_version

def test_version_generation():
    version = (0, 7, 3)
    assert generate_version(version, prerelease=None, revision=None) == '0.7.3'

def test_version_generation_with_prerelease_revision():
    version = (0, 7, 3)
    prerelease = 'alpha'
    revision = 2
    assert generate_version(version, prerelease=prerelease, revision=revision) == '0.7.3-alpha.2'