def test_imports():

    from camelot.backends.pdfium_backend import PdfiumBackend  # noqa: F401
    from camelot.backends.poppler_backend import PopplerBackend  # noqa: F401
    from camelot.handlers import Hybrid  # noqa: F401
    from camelot.handlers import Lattice  # noqa: F401
    from camelot.handlers import Network  # noqa: F401
    from camelot.handlers import Stream  # noqa: F401
    from pypdf_table_extraction.backends.poppler_backend import (  # noqa: F401,F811
        PopplerBackend,
    )
    from pypdf_table_extraction.parsers import Hybrid  # noqa: F401, F811
    from pypdf_table_extraction.parsers import Lattice  # noqa: F401, F811
    from pypdf_table_extraction.parsers import Network  # noqa: F401, F811
    from pypdf_table_extraction.parsers import Stream  # noqa: F401, F811
