<p align="center">
   <img src="https://github.com/py-pdf/pypdf_table_extraction/blob/main/docs/_static/pypdf-table-extraction.png" width="200">
</p>

# pypdf_table_extraction (Camelot): PDF Table Extraction for Humans

[![tests](https://github.com/py-pdf/pypdf_table_extraction/actions/workflows/tests.yml/badge.svg)](https://github.com/py-pdf/pypdf_table_extraction/actions/workflows/tests.yml) [![Documentation Status](https://readthedocs.org/projects/pypdf-table-extraction/badge/?version=latest)](https://pypdf-table-extraction.readthedocs.io/en/latest/)
[![codecov.io](https://codecov.io/github/py-pdf/pypdf_table_extraction/badge.svg?branch=main&service=github)](https://codecov.io/github/py-pdf/pypdf_table_extraction/?branch=main)
[![image](https://img.shields.io/pypi/v/pypdf-table-extraction.svg)](https://pypi.org/project/pypdf-table-extraction/) [![image](https://img.shields.io/pypi/l/pypdf-table-extraction.svg)](https://pypi.org/project/pypdf-table-extraction/) [![image](https://img.shields.io/pypi/pyversions/pypdf-table-extraction.svg)](https://pypi.org/project/pypdf-table-extraction/)

**pypdf_table_extraction** Formerly known as [Camelot](https://github.com/camelot-dev/camelot) is a Python library that can help you extract tables from PDFs!

---

**Here's how you can extract tables from PDFs.**
You can check out the quickstart notebook. [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-pdf/pypdf_table_extraction/blob/main/examples/pypdf_table_extraction_quick_start_notebook.ipynb)

Or follow the example below.
You can check out the PDF used in this example [here](https://github.com/py-pdf/pypdf_table_extraction/blob/main/docs/_static/pdf/foo.pdf).

```python3
>>> import pypdf_table_extraction
>>> tables = pypdf_table_extraction.read_pdf('foo.pdf')
>>> tables
<TableList n=1>
>>> tables.export('foo.csv', f='csv', compress=True) # json, excel, html, markdown, sqlite
>>> tables[0]
<Table shape=(7, 7)>
>>> tables[0].parsing_report
{
    'accuracy': 99.02,
    'whitespace': 12.24,
    'order': 1,
    'page': 1
}
>>> tables[0].to_csv('foo.csv') # to_json, to_excel, to_html, to_markdown, to_sqlite
>>> tables[0].df # get a pandas DataFrame!
```

| Cycle Name | KI (1/km) | Distance (mi) | Percent Fuel Savings |                 |                 |                |
| ---------- | --------- | ------------- | -------------------- | --------------- | --------------- | -------------- |
|            |           |               | Improved Speed       | Decreased Accel | Eliminate Stops | Decreased Idle |
| 2012_2     | 3.30      | 1.3           | 5.9%                 | 9.5%            | 29.2%           | 17.4%          |
| 2145_1     | 0.68      | 11.2          | 2.4%                 | 0.1%            | 9.5%            | 2.7%           |
| 4234_1     | 0.59      | 58.7          | 8.5%                 | 1.3%            | 8.5%            | 3.3%           |
| 2032_2     | 0.17      | 57.8          | 21.7%                | 0.3%            | 2.7%            | 1.2%           |
| 4171_1     | 0.07      | 173.9         | 58.1%                | 1.6%            | 2.1%            | 0.5%           |

pypdf_table_extraction also comes packaged with a [command-line interface](https://pypdf-table-extraction.readthedocs.io/en/latest/user/cli.html)!

Refer to the [QuickStart Guide](https://github.com/py-pdf/pypdf_table_extraction/blob/main/docs/user/quickstart.rst#quickstart) to quickly get started with pypdf_table_extraction, extract tables from PDFs and explore some basic options.

**Tip:** Visit the `parser-comparison-notebook` to get an overview of all the packed parsers and their features. [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-pdf/pypdf_table_extraction/blob/main/examples/parser-comparison-notebook.ipynb)

**Note:** pypdf_table_extraction only works with text-based PDFs and not scanned documents. (As Tabula [explains](https://github.com/tabulapdf/tabula#why-tabula), "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

You can check out some frequently asked questions [here](https://pypdf-table-extraction.readthedocs.io/en/latest/user/faq.html).

## Why pypdf_table_extraction?

- **Configurability**: pypdf_table_extraction gives you control over the table extraction process with [tweakable settings](https://pypdf-table-extraction.readthedocs.io/en/latest/user/advanced.html).
- **Metrics**: You can discard bad tables based on metrics like accuracy and whitespace, without having to manually look at each table.
- **Output**: Each table is extracted into a **pandas DataFrame**, which seamlessly integrates into [ETL and data analysis workflows](https://gist.github.com/vinayak-mehta/e5949f7c2410a0e12f25d3682dc9e873). You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML, Markdown, and Sqlite.

See [comparison with similar libraries and tools](https://github.com/py-pdf/pypdf_table_extraction/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools).

## Installation

### Using conda

The easiest way to install pypdf_table_extraction is with [conda](https://conda.io/docs/), which is a package manager and environment management system for the [Anaconda](http://docs.continuum.io/anaconda/) distribution.

```bash
conda install -c conda-forge pypdf-table-extraction
```

### Using pip

After [installing the dependencies](https://pypdf-table-extraction.readthedocs.io/en/latest/user/install-deps.html) ([tk](https://packages.ubuntu.com/bionic/python/python-tk) and [ghostscript](https://www.ghostscript.com/)), you can also just use pip to install pypdf_table_extraction:

```bash
pip install pypdf-table-extraction
```

### From the source code

After [installing the dependencies](https://pypdf-table-extraction.readthedocs.io/en/latest/user/install.html#using-pip), clone the repo using:

```bash
git clone https://github.com/py-pdf/pypdf_table_extraction.git
```

and install using pip:

```
cd pypdf_table_extraction
pip install "."
```

## Documentation

The documentation is available at [http://pypdf-table-extraction.readthedocs.io/](http://pypdf-table-extraction.readthedocs.io/).

## Wrappers

- [camelot-php](https://github.com/randomstate/camelot-php) provides a [PHP](https://www.php.net/) wrapper on Camelot.

## Related projects

- [camelot-sharp](https://github.com/BobLd/camelot-sharp) provides a C sharp implementation of pypdf_table_extraction (Camelot).

## Contributing

The [Contributor's Guide](https://pypdf-table-extraction.readthedocs.io/en/latest/dev/contributing.html) has detailed information about contributing issues, documentation, code, and tests.

## Versioning

pypdf_table_extraction uses [Semantic Versioning](https://semver.org/). For the available versions, see the tags on this repository. For the changelog, you can check out the [releases](https://github.com/py-pdf/pypdf_table_extraction/releases) page.

## License

This project is licensed under the MIT License, see the [LICENSE](https://github.com/py-pdf/pypdf_table_extraction/blob/main/LICENSE) file for details.
