def test_imports():

    from camelot.backends.poppler_backend import PopplerBackend  # noqa: F401
    from camelot.handlers import Lattice  # noqa: F401
    from camelot.handlers import Stream  # noqa: F401
    from pypdf_table_extraction.backends.poppler_backend import (  # noqa: F401,F811
        PopplerBackend,
    )
    from pypdf_table_extraction.parsers import Lattice  # noqa: F401, F811
    from pypdf_table_extraction.parsers import Stream  # noqa: F401, F811
