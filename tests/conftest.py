import os
import sys

import pytest


skip_on_windows = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Ghostscript not installed in Windows test environment",
)

skip_pdftopng = pytest.mark.skip(
    reason="Ghostscript not installed in Windows test environment",
)


@pytest.fixture
def testdir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")


@pytest.fixture
def foo_pdf(testdir):
    return os.path.join(testdir, "foo.pdf")
