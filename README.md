<p align="center">
  <img src="https://raw.githubusercontent.com/camelot-dev/camelot/master/docs/_static/camelot.png" width="200">
</p>

# Camelot: PDF Table Extraction for Humans

[![tests](https://github.com/camelot-dev/camelot/actions/workflows/tests.yml/badge.svg)](https://github.com/camelot-dev/camelot/actions/workflows/tests.yml) [![Documentation Status](https://readthedocs.org/projects/camelot-py/badge/?version=latest)](https://camelot-py.readthedocs.io/en/latest/)
[![codecov.io](https://codecov.io/github/camelot-dev/camelot/badge.svg?branch=master&service=github)](https://codecov.io/github/camelot-dev/camelot?branch=master)
[![image](https://img.shields.io/pypi/v/camelot-py.svg)](https://pypi.org/project/camelot-py/) [![image](https://img.shields.io/pypi/l/camelot-py.svg)](https://pypi.org/project/camelot-py/) [![image](https://img.shields.io/pypi/pyversions/camelot-py.svg)](https://pypi.org/project/camelot-py/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**Camelot** is a Python library that can help you extract tables from PDFs.

## Features

- 📊 **Five parsers** — `lattice` (ruled tables), `stream` (whitespace), the text-alignment `network` / `hybrid`, and the optional neural `ml` (Table Transformer) for hard borderless tables — plus `flavor="auto"` to pick one for you.
- 🤖 **Borderless & scanned** — the optional `ml` backend (`pip install "camelot-py[ml]"`) recovers structure that the heuristic parsers can't on borderless tables; add `[ocr]` to read **scanned / image-only PDFs** with no text layer.
- 🧠 **Vector + raster line detection** — `engine="combined"` unions the PDF's native vector ruled lines with OpenCV detection, so faintly-ruled tables are still found.
- 🐼 **pandas output** — every table is a `DataFrame`, ready for analysis.
- 📤 **Many export formats** — CSV, JSON, Excel, HTML, Markdown, and SQLite.
- 📐 **Quality metrics** — accuracy, whitespace, and a confidence score per table; drop noise with `TableList.filter(...)`.
- 🧩 **Multi-page tables** — stitch continuations across pages with `stack_contiguous()`.
- 🎛️ **Highly configurable** — table areas/regions, column separators, text processing, and more.
- 🔌 **Flexible input** — a file path, URL, raw `bytes`, or any binary file-like object.
- 🖥️ **CLI included** — `camelot lattice file.pdf`, etc.
- 📦 **Light install** — the default pdfium backend is bundled, with no system dependencies.

---

**Extract tables from PDFs in just a few lines of code:**

Try it yourself in our interactive quickstart notebook. [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camelot-dev/camelot/blob/master/examples/camelot-quickstart-notebook.ipynb)

Or check out a simple example using [this pdf](https://github.com/camelot-dev/camelot/blob/master/docs/_static/pdf/foo.pdf).

<pre>
>>> import camelot
>>> tables = camelot.read_pdf('foo.pdf')
>>> tables
&lt;TableList n=1&gt;
>>> tables.export('foo.csv', f='csv', compress=True) # json, excel, html, markdown, sqlite
>>> tables[0]
&lt;Table shape=(7, 7)&gt;
>>> tables[0].parsing_report
{
    'accuracy': 99.02,
    'whitespace': 12.24,
    'order': 1,
    'page': 1
}
>>> tables[0].to_csv('foo.csv') # to_json, to_excel, to_html, to_markdown, to_sqlite
>>> tables[0].df # get a pandas DataFrame!
</pre>

| Cycle Name | KI (1/km) | Distance (mi) | Percent Fuel Savings |                 |                 |                |
| ---------- | --------- | ------------- | -------------------- | --------------- | --------------- | -------------- |
|            |           |               | Improved Speed       | Decreased Accel | Eliminate Stops | Decreased Idle |
| 2012_2     | 3.30      | 1.3           | 5.9%                 | 9.5%            | 29.2%           | 17.4%          |
| 2145_1     | 0.68      | 11.2          | 2.4%                 | 0.1%            | 9.5%            | 2.7%           |
| 4234_1     | 0.59      | 58.7          | 8.5%                 | 1.3%            | 8.5%            | 3.3%           |
| 2032_2     | 0.17      | 57.8          | 21.7%                | 0.3%            | 2.7%            | 1.2%           |
| 4171_1     | 0.07      | 173.9         | 58.1%                | 1.6%            | 2.1%            | 0.5%           |

Camelot also comes packaged with a [command-line interface](https://camelot-py.readthedocs.io/en/latest/user/cli.html)!

Refer to the [QuickStart Guide](https://github.com/camelot-dev/camelot/blob/master/docs/user/quickstart.rst#quickstart) to quickly get started with Camelot, extract tables from PDFs and explore some basic options.

**Tip:** Visit the `parser-comparison-notebook` to get an overview of all the packed parsers and their features. [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camelot-dev/camelot/blob/master/examples/parser-comparison-notebook.ipynb)

**Note:** The built-in parsers need a text-based PDF (as Tabula [explains](https://github.com/tabulapdf/tabula#why-tabula), "If you can click and drag to select text in your table in a PDF viewer, then your PDF is text-based"). For **scanned / image-only** PDFs, install the neural backend with OCR — `pip install "camelot-py[ml,ocr]"` — and use `camelot.read_pdf(..., flavor="ml")`: the model reads the structure from the page image and OCR supplies the text.

You can check out some frequently asked questions [here](https://camelot-py.readthedocs.io/en/latest/user/faq.html).

## Which parser should I use?

| Your PDF                                      | Use                                                | Why                                                                                                                                                               |
| --------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ruled tables** (lines between cells)        | `flavor="lattice"` (default)                       | Deterministic; detects the grid from the ruled lines. `engine="combined"` also catches faint vector rules.                                                        |
| **Borderless tables** (whitespace-separated)  | `flavor="network"` or `"stream"`                   | Text-alignment / whitespace heuristics — fast, no extra dependencies.                                                                                             |
| **Borderless tables, best quality**           | `flavor="ml"` (`pip install "camelot-py[ml]"`)     | A Table Transformer model recovers structure heuristics can't — on FinTabNet it roughly doubles borderless TEDS vs `network`/`hybrid`. Heavier (PyTorch); opt-in. |
| **Scanned / image-only PDFs** (no text layer) | `flavor="ml"` + `pip install "camelot-py[ml,ocr]"` | Structure from the model, text from OCR.                                                                                                                          |
| **Mixed / not sure**                          | `flavor="auto"`                                    | Picks `lattice` or `network` per page.                                                                                                                            |

The `ml` backend keeps Camelot honest: the model only supplies the table
**structure**, while cell **text** comes from the PDF's own text layer (or OCR
for scans) — so it never invents or alters a value.

## Why Camelot?

- **Configurability**: Camelot gives you control over the table extraction process with [tweakable settings](https://camelot-py.readthedocs.io/en/latest/user/advanced.html).
- **Metrics**: You can discard bad tables based on metrics like accuracy and whitespace, without having to manually look at each table.
- **Output**: Each table is extracted into a **pandas DataFrame**, which seamlessly integrates into [ETL and data analysis workflows](https://gist.github.com/vinayak-mehta/e5949f7c2410a0e12f25d3682dc9e873). You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML, Markdown, and Sqlite.

See [comparison with similar libraries and tools](https://camelot-py.readthedocs.io/en/latest/user/comparison.html).

## Installation

Camelot's default image-conversion backend is [pdfium](https://pypi.org/project/pypdfium2/), which ships as a wheel — so a plain install needs **no system dependencies**. The optional [ghostscript](https://www.ghostscript.com/) and poppler backends require [additional dependencies](https://camelot-py.readthedocs.io/en/latest/user/install-deps.html).

### Using uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. To add Camelot to a project:

```bash
uv add camelot-py
```

Or to install it into the current environment:

```bash
uv pip install camelot-py
```

### Using pip

```bash
pip install "camelot-py"
```

### Using conda

[conda](https://conda.io/docs/) is the package manager for the [Anaconda](http://docs.continuum.io/anaconda/) distribution:

```bash
conda install -c conda-forge camelot-py
```

### From the source code

```bash
git clone https://github.com/camelot-dev/camelot.git
cd camelot
uv pip install "."  # or: pip install "."
```

### Optional extras

```bash
pip install "camelot-py[ml]"      # neural flavor='ml' (Table Transformer; pulls PyTorch)
pip install "camelot-py[ocr]"     # OCR text source for scanned PDFs (use with [ml])
pip install "camelot-py[ml,ocr]"  # both — borderless + scanned
pip install "camelot-py[plot]"    # matplotlib debug plots
```

The core install stays light: `[ml]`/`[ocr]` are imported lazily, so a plain
`import camelot` never loads PyTorch or OCR.

## Documentation

The documentation is available at [http://camelot-py.readthedocs.io/](http://camelot-py.readthedocs.io/).

## Contributing

The [Contributor's Guide](https://camelot-py.readthedocs.io/en/latest/dev/contributing.html) has detailed information about contributing issues, documentation, code, and tests.

## Versioning

Camelot uses [Semantic Versioning](https://semver.org/). For the available versions, see the tags on this repository. For the changelog, you can check out the [releases](https://github.com/camelot-dev/camelot/releases) page.

## License

This project is licensed under the MIT License, see the [LICENSE](https://github.com/camelot-dev/camelot/blob/master/LICENSE) file for details.

The documentation theme is licensed under a seperate BSD-like License, see the [LICENSE](https://github.com/camelot-dev/camelot/blob/master/docs/_themes/LICENSE) file for details.
