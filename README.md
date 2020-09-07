<p align="center">
   <img src="https://raw.githubusercontent.com/camelot-dev/camelot/master/docs/_static/camelot.png" width="200">
</p>

# Camelot: PDF Table Extraction for Humans

[![Build Status](https://travis-ci.org/camelot-dev/camelot.svg?branch=master)](https://travis-ci.org/camelot-dev/camelot) [![Documentation Status](https://readthedocs.org/projects/camelot-py/badge/?version=master)](https://camelot-py.readthedocs.io/en/master/)
 [![codecov.io](https://codecov.io/github/camelot-dev/camelot/badge.svg?branch=master&service=github)](https://codecov.io/github/camelot-dev/camelot?branch=master)
 [![image](https://img.shields.io/pypi/v/camelot-py.svg)](https://pypi.org/project/camelot-py/) [![image](https://img.shields.io/pypi/l/camelot-py.svg)](https://pypi.org/project/camelot-py/) [![image](https://img.shields.io/pypi/pyversions/camelot-py.svg)](https://pypi.org/project/camelot-py/) [![Gitter chat](https://badges.gitter.im/camelot-dev/Lobby.png)](https://gitter.im/camelot-dev/Lobby)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![image](https://img.shields.io/badge/continous%20quality-deepsource-lightgrey)](https://deepsource.io/gh/camelot-dev/camelot/?ref=repository-badge)


**Camelot** is a Python library that can help you extract tables from PDFs!

**Note:** You can also check out [Excalibur](https://github.com/camelot-dev/excalibur), the web interface to Camelot!

---

**Here's how you can extract tables from PDFs.** You can check out the PDF used in this example [here](https://github.com/camelot-dev/camelot/blob/master/docs/_static/pdf/foo.pdf).

<pre>
>>> import camelot
>>> tables = camelot.read_pdf('foo.pdf')
>>> tables
&lt;TableList n=1&gt;
>>> tables.export('foo.csv', f='csv', compress=True) # json, excel, html, sqlite
>>> tables[0]
&lt;Table shape=(7, 7)&gt;
>>> tables[0].parsing_report
{
    'accuracy': 99.02,
    'whitespace': 12.24,
    'order': 1,
    'page': 1
}
>>> tables[0].to_csv('foo.csv') # to_json, to_excel, to_html, to_sqlite
>>> tables[0].df # get a pandas DataFrame!
</pre>

| Cycle Name | KI (1/km) | Distance (mi) | Percent Fuel Savings |                 |                 |                |
|------------|-----------|---------------|----------------------|-----------------|-----------------|----------------|
|            |           |               | Improved Speed       | Decreased Accel | Eliminate Stops | Decreased Idle |
| 2012_2     | 3.30      | 1.3           | 5.9%                 | 9.5%            | 29.2%           | 17.4%          |
| 2145_1     | 0.68      | 11.2          | 2.4%                 | 0.1%            | 9.5%            | 2.7%           |
| 4234_1     | 0.59      | 58.7          | 8.5%                 | 1.3%            | 8.5%            | 3.3%           |
| 2032_2     | 0.17      | 57.8          | 21.7%                | 0.3%            | 2.7%            | 1.2%           |
| 4171_1     | 0.07      | 173.9         | 58.1%                | 1.6%            | 2.1%            | 0.5%           |

Camelot also comes packaged with a [command-line interface](https://camelot-py.readthedocs.io/en/master/user/cli.html)!

**Note:** Camelot only works with text-based PDFs and not scanned documents. (As Tabula [explains](https://github.com/tabulapdf/tabula#why-tabula), "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based".)

## Why Camelot?

- **Configurability**: Camelot gives you control over the table extraction process with its [tweakable settings](https://camelot-py.readthedocs.io/en/master/user/advanced.html).
- **Metrics**: Bad tables can be discarded based on metrics like accuracy and whitespace, without having to manually look at each table.
- **Output**: Each table is extracted into a **pandas DataFrame**, which seamlessly integrates into [ETL and data analysis workflows](https://gist.github.com/vinayak-mehta/e5949f7c2410a0e12f25d3682dc9e873). You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML and Sqlite.

See [comparison with similar libraries and tools](https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools).

## Support the development

If Camelot has helped you, please consider supporting its development with a one-time or monthly donation [on OpenCollective](https://opencollective.com/camelot).

## Installation

### Using conda

The easiest way to install Camelot is with [conda](https://conda.io/docs/), which is a package manager and environment management system for the [Anaconda](http://docs.continuum.io/anaconda/) distribution.

<pre>
$ conda install -c conda-forge camelot-py
</pre>

### Using pip

After [installing the dependencies](https://camelot-py.readthedocs.io/en/master/user/install-deps.html) ([tk](https://packages.ubuntu.com/bionic/python/python-tk) and [ghostscript](https://www.ghostscript.com/)), you can also just use pip to install Camelot:

<pre>
$ pip install "camelot-py[cv]"
</pre>

### From the source code

After [installing the dependencies](https://camelot-py.readthedocs.io/en/master/user/install.html#using-pip), clone the repo using:

<pre>
$ git clone https://www.github.com/camelot-dev/camelot
</pre>

and install Camelot using pip:

<pre>
$ cd camelot
$ pip install ".[cv]"
</pre>

## Documentation

The documentation is available at [http://camelot-py.readthedocs.io/](http://camelot-py.readthedocs.io/).

## Wrappers

- [camelot-php](https://github.com/randomstate/camelot-php) provides a [PHP](https://www.php.net/) wrapper on Camelot.

## Contributing

The [Contributor's Guide](https://camelot-py.readthedocs.io/en/master/dev/contributing.html) has detailed information about contributing issues, documentation, code, and tests.

## Versioning

Camelot uses [Semantic Versioning](https://semver.org/). For the available versions, see the tags on this repository. For the changelog, you can check out [HISTORY.md](https://github.com/camelot-dev/camelot/blob/master/HISTORY.md).

## License

This project is licensed under the MIT License, see the [LICENSE](https://github.com/camelot-dev/camelot/blob/master/LICENSE) file for details.
