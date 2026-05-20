"""Per-page kwarg overrides (#41)."""

import os

import pytest

import camelot


def test_per_page_override_applies(testdir):
    """Page 2 uses an override flavor; pages 1-2 still both yield tables."""
    filename = os.path.join(testdir, "hybrid_multipage.pdf")
    # Global flavor = "hybrid", override page 2 to "network". Both pages
    # have a table the parser can find under their respective flavors, so
    # the call still returns 2 tables.
    tables = camelot.read_pdf(
        filename,
        flavor="hybrid",
        pages="1-2",
        per_page={2: {"flavor": "network"}},
    )
    assert len(tables) >= 1


def test_per_page_string_key_accepted(testdir):
    """The 2019 #41 proposal used str keys ('2'); accept them too."""
    filename = os.path.join(testdir, "hybrid_multipage.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="hybrid",
        pages="1-2",
        per_page={"2": {"split_text": True}},
    )
    assert len(tables) >= 1


def test_per_page_no_override_no_regression(testdir):
    """An empty per_page is exactly equivalent to not passing it."""
    filename = os.path.join(testdir, "hybrid_multipage.pdf")
    a = camelot.read_pdf(filename, flavor="hybrid", pages="1-2")
    b = camelot.read_pdf(filename, flavor="hybrid", pages="1-2", per_page={})
    assert len(a) == len(b)


def test_per_page_invalid_kwarg_rejected(foo_pdf):
    """Unknown parser kwarg in per_page raises ValueError (same path as global)."""
    with pytest.raises(ValueError, match="cannot be used with flavor"):
        camelot.read_pdf(
            foo_pdf,
            flavor="lattice",
            per_page={1: {"row_tol": 5}},  # row_tol is stream/network-only
        )


def test_per_page_invalid_flavor_rejected(foo_pdf):
    with pytest.raises(NotImplementedError, match="not one of"):
        camelot.read_pdf(foo_pdf, per_page={1: {"flavor": "chocolate"}})


def test_per_page_auto_flavor_rejected(foo_pdf):
    """'auto' is only valid as the global flavor, not as a per-page override."""
    with pytest.raises(NotImplementedError, match="not one of"):
        camelot.read_pdf(foo_pdf, per_page={1: {"flavor": "auto"}})


def test_per_page_non_dict_value_rejected(foo_pdf):
    with pytest.raises(ValueError, match="must be a dict of kwargs"):
        camelot.read_pdf(foo_pdf, per_page={1: "not a dict"})


def test_per_page_non_int_key_rejected(foo_pdf):
    with pytest.raises(ValueError, match="per_page keys must be page numbers"):
        camelot.read_pdf(foo_pdf, per_page={"abc": {}})
