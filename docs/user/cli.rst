.. _cli:

Command-Line Interface
======================

pypdf-table-extraction comes with a command-line interface.

You can print the help for the interface by typing ``camelot --help`` in your favorite terminal program, as shown below. Furthermore, you can print the help for each command by typing ``pypdf-table-extraction <command> --help``. Try it out!

:: console

  Usage: pypdf-table-extraction [OPTIONS] COMMAND [ARGS]...

    pypdf-table-extraction: PDF Table Extraction for Humans

  Options:
    --version                       Show the version and exit.
    -q, --quiet TEXT                Suppress logs and warnings.
    -p, --pages TEXT                Comma-separated page numbers. Example: 1,3,4
                                    or 1,4-end.
    -pw, --password TEXT            Password for decryption.
    -o, --output TEXT               Output file path.
    -f, --format [csv|json|excel|html]
                                    Output file format.

    -z, --zip                       Create ZIP archive.
    -split, --split_text            Split text that spans across multiple cells.
    -flag, --flag_size              Flag text based on font size. Useful to
                                    detect super/subscripts.
    -strip, --strip_text            Characters that should be stripped from a
                                    string before assigning it to a cell.

    -M, --margins                   <FLOAT FLOAT FLOAT>...
                                        PDFMiner char_margin, line_margin and
                                        word_margin.

    --help                          Show this message and exit.

  Commands:
    lattice  Use lines between text to parse the table.
    stream   Use spaces between text to parse the table.

