import pytest

from camelot.backends.base import ConversionBackend


def test_conversion_backend_methods_are_abstract():
    # The base class defines the interface; both methods must raise until a
    # concrete backend overrides them.
    backend = ConversionBackend()
    with pytest.raises(NotImplementedError):
        backend.installed()
    with pytest.raises(NotImplementedError):
        backend.convert("in.pdf", "out.png")
