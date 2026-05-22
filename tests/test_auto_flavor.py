"""Regression tests for the flavor='auto' detection probe (_detect_flavor)."""

import camelot
from camelot.io import _detect_flavor


def test_detect_flavor_returns_lattice_for_ruled_pdf(foo_pdf):
    # foo.pdf is a ruled (lattice) table. The probe must classify it as
    # 'lattice'. Regression: _detect_flavor passed a non-existent
    # `resolution=` kwarg to backend.convert(), which raised TypeError that
    # the bare except swallowed, so 'auto' silently fell back to 'network'
    # for *every* PDF.
    assert _detect_flavor(foo_pdf) == "lattice"


def test_auto_flavor_extracts_ruled_table(foo_pdf):
    # End-to-end: flavor='auto' on a ruled PDF should pick lattice and
    # extract the table (not mis-route to network).
    tables = camelot.read_pdf(foo_pdf, flavor="auto")
    assert len(tables) == 1
    assert tables[0].shape[0] >= 2 and tables[0].shape[1] >= 2
