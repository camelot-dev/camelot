Release History
===============

master
------

0.11.0 (2023-02-26)
------------------

- Replace `PdfFileReader` with `PdfReader` and pin PyPDF to `>=3.0.0`. [#307](https://github.com/camelot-dev/camelot/pull/307) by [Martin Thoma](https://github.com/MartinThoma).


0.10.1 (2021-07-11)
------------------

- Change extra requirements from `cv` to `base`. You can use `pip install "camelot-py[base]"` to install everything required to run camelot.

0.10.0 (2021-07-11)
------------------

**Improvements**

- Add support for multiple image conversion backends. [#198](https://github.com/camelot-dev/camelot/pull/198) and [#253](https://github.com/camelot-dev/camelot/pull/253) by Vinayak Mehta.
- Add markdown export format. [#222](https://github.com/camelot-dev/camelot/pull/222/) by [Lucas Cimon](https://github.com/Lucas-C).

**Documentation**

- Add faq section. [#216](https://github.com/camelot-dev/camelot/pull/216) by [Stefano Fiorucci](https://github.com/anakin87).

0.9.0 (2021-06-15)
------------------

**Bugfixes**

- Fix use of resolution argument to generate image with ghostscript. [#231](https://github.com/camelot-dev/camelot/pull/231) by [Tiago Samaha Cordeiro](https://github.com/tiagosamaha).
- [#15](https://github.com/camelot-dev/camelot/issues/15) Fix duplicate strings being assigned to the same cell. [#206](https://github.com/camelot-dev/camelot/pull/206) by [Eduardo Gonzalez Lopez de Murillas](https://github.com/edugonza).
- Save plot when filename is specified. [#121](https://github.com/camelot-dev/camelot/pull/121) by [Jens Diemer](https://github.com/jedie).
- Close file streams explicitly. [#202](https://github.com/camelot-dev/camelot/pull/202) by [Martin Abente Lahaye](https://github.com/tchx84).
- Use correct re.sub signature. [#186](https://github.com/camelot-dev/camelot/pull/186) by [pevisscher](https://github.com/pevisscher).
- [#183](https://github.com/camelot-dev/camelot/issues/183) Fix UnicodeEncodeError when using Stream flavor by adding encoding kwarg to `to_html`. [#188](https://github.com/camelot-dev/camelot/pull/188) by [Stefano Fiorucci](https://github.com/anakin87).
- [#179](https://github.com/camelot-dev/camelot/issues/179) Fix `max() arg is an empty sequence` error on PDFs with blank pages. [#189](https://github.com/camelot-dev/camelot/pull/189) by Vinayak Mehta.

**Improvements**

- Add `line_overlap` and `boxes_flow` to `LAParams`. [#219](https://github.com/camelot-dev/camelot/pull/219) by [Arnie97](https://github.com/Arnie97).
- [Add bug report template.](https://github.com/camelot-dev/camelot/commit/0a3944e54d133b701edfe9c7546ff11289301ba8)
- Move from [Travis to GitHub Actions](https://github.com/camelot-dev/camelot/pull/241).
- Update `.readthedocs.yml` and [remove requirements.txt](https://github.com/camelot-dev/camelot/commit/7ab5db39d07baa4063f975e9e00f6073340e04c1#diff-cde814ef2f549dc093f5b8fc533b7e8f47e7b32a8081e0760e57d5c25a1139d9)

**Documentation**

- [#193](https://github.com/camelot-dev/camelot/issues/193) Add better checks to confirm proper installation of ghostscript. [#196](https://github.com/camelot-dev/camelot/pull/196) by [jimhall](https://github.com/jimhall).
- Update `advanced.rst` plotting examples. [#119](https://github.com/camelot-dev/camelot/pull/119) by [Jens Diemer](https://github.com/jedie).

0.8.2 (2020-07-27)
------------------

* Revert the changes in `0.8.1`.

0.8.1 (2020-07-21)
------------------

**Bugfixes**

* [#169](https://github.com/camelot-dev/camelot/issues/169) Fix import error caused by `pdfminer.six==20200720`. [#171](https://github.com/camelot-dev/camelot/pull/171) by Vinayak Mehta.

0.8.0 (2020-05-24)
------------------

**Improvements**

* Drop Python 2 support!
    * Remove Python 2.7 and 3.5 support.
    * Replace all instances of `.format` with f-strings.
    * Remove all `__future__` imports.
    * Fix HTTP 403 forbidden exception in read_pdf(url) and remove Python 2 urllib support.
    * Fix test data.

**Bugfixes**

* Fix library discovery on Windows. [#32](https://github.com/camelot-dev/camelot/pull/32) by [KOLANICH](https://github.com/KOLANICH).
* Fix calling convention of callback functions. [#34](https://github.com/camelot-dev/camelot/pull/34) by [KOLANICH](https://github.com/KOLANICH).

0.7.3 (2019-07-07)
------------------

**Improvements**

* Camelot now follows the Black code style! [#1](https://github.com/camelot-dev/camelot/pull/1) and [#3](https://github.com/camelot-dev/camelot/pull/3).

**Bugfixes**

* Fix Click.HelpFormatter monkey-patch. [#5](https://github.com/camelot-dev/camelot/pull/5) by [Dimiter Naydenov](https://github.com/dimitern).
* Fix strip_text argument getting ignored. [#4](https://github.com/camelot-dev/camelot/pull/4) by [Dimiter Naydenov](https://github.com/dimitern).
* [#25](https://github.com/camelot-dev/camelot/issues/25) edge_tol skipped in read_pdf. [#26](https://github.com/camelot-dev/camelot/pull/26) by Vinayak Mehta.
* Fix pytest deprecation warning. [#2](https://github.com/camelot-dev/camelot/pull/2) by Vinayak Mehta.
* [#293](https://github.com/socialcopsdev/camelot/issues/293) Split text ignores all text to the right of last cut. [#294](https://github.com/socialcopsdev/camelot/pull/294) by Vinayak Mehta.
* [#277](https://github.com/socialcopsdev/camelot/issues/277) Sort TableList by order of tables in PDF. [#283](https://github.com/socialcopsdev/camelot/pull/283) by [Sym Roe](https://github.com/symroe).
* [#312](https://github.com/socialcopsdev/camelot/issues/312) `table_regions` throws `ValueError` when `flavor='stream'`. [#332](https://github.com/socialcopsdev/camelot/pull/332) by Vinayak Mehta.

0.7.2 (2019-01-10)
------------------

**Bugfixes**

* [#245](https://github.com/socialcopsdev/camelot/issues/245) Fix AttributeError for encrypted files. [#251](https://github.com/socialcopsdev/camelot/pull/251) by Yatin Taluja.


0.7.1 (2019-01-06)
------------------

**Bugfixes**

* Move ghostscript import to inside the function so Anaconda builds don't fail.

0.7.0 (2019-01-05)
------------------

**Improvements**

* [#240](https://github.com/socialcopsdev/camelot/issues/209) Add support to analyze only certain page regions to look for tables. [#243](https://github.com/socialcopsdev/camelot/pull/243) by Vinayak Mehta.
    * You can use `table_regions` in `read_pdf()` to specify approximate page regions which may contain tables.
    * Kwarg `line_size_scaling` is now called `line_scale`.
* [#212](https://github.com/socialcopsdev/camelot/issues/212) Add support to export as sqlite database. [#244](https://github.com/socialcopsdev/camelot/pull/244) by Vinayak Mehta.
* [#239](https://github.com/socialcopsdev/camelot/issues/239) Raise warning if PDF is image-based. [#240](https://github.com/socialcopsdev/camelot/pull/240) by Vinayak Mehta.

**Documentation**

* Remove mention of old mesh kwarg from docs. [#241](https://github.com/socialcopsdev/camelot/pull/241) by [fte10kso](https://github.com/fte10kso).

**Note**: The python wrapper to Ghostscript's C API is now vendorized under the `ext` module. This was done due to unavailability of the [ghostscript](https://pypi.org/project/ghostscript/) package on Anaconda. The code should be removed after we submit a recipe for it to conda-forge. With this release, the user doesn't need to ensure that the Ghostscript executable is available on the PATH variable.

0.6.0 (2018-12-24)
------------------

**Improvements**

* [#91](https://github.com/socialcopsdev/camelot/issues/91) Add support to read from url. [#236](https://github.com/socialcopsdev/camelot/pull/236) by Vinayak Mehta.
* [#229](https://github.com/socialcopsdev/camelot/issues/229), [#230](https://github.com/socialcopsdev/camelot/issues/230) and [#233](https://github.com/socialcopsdev/camelot/issues/233) New configuration parameters. [#234](https://github.com/socialcopsdev/camelot/pull/234) by Vinayak Mehta.
    * `strip_text`: To define characters that should be stripped from each string.
    * `edge_tol`: Tolerance parameter for extending textedges vertically.
    * `resolution`: Resolution used for PDF to PNG conversion.
    * Check out the [advanced docs](https://camelot-py.readthedocs.io/en/master/user/advanced.html#strip-characters-from-text) for usage details.
* [#170](https://github.com/socialcopsdev/camelot/issues/170) Add option to pass pdfminer layout kwargs. [#232](https://github.com/socialcopsdev/camelot/pull/232) by Vinayak Mehta.
    * Keyword arguments for [pdfminer.layout.LAParams](https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33) can now be passed using `layout_kwargs` in `read_pdf()`.
    * The `margins` keyword argument in `read_pdf()` is now deprecated.

0.5.0 (2018-12-13)
------------------

**Improvements**

* [#207](https://github.com/socialcopsdev/camelot/issues/207) Add a plot type for Stream text edges and detected table areas. [#224](https://github.com/socialcopsdev/camelot/pull/224) by Vinayak Mehta.
* [#204](https://github.com/socialcopsdev/camelot/issues/204) `suppress_warnings` is now called `suppress_stdout`. [#225](https://github.com/socialcopsdev/camelot/pull/225) by Vinayak Mehta.

**Bugfixes**

* [#217](https://github.com/socialcopsdev/camelot/issues/217) Fix IndexError when scale is large.
* [#105](https://github.com/socialcopsdev/camelot/issues/105), [#192](https://github.com/socialcopsdev/camelot/issues/192) and [#215](https://github.com/socialcopsdev/camelot/issues/215) in [#227](https://github.com/socialcopsdev/camelot/pull/227) by Vinayak Mehta.

**Documentation**

* Add pdfplumber comparison and update Tabula (stream) comparison. Check out the [wiki page](https://github.com/socialcopsdev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools).

0.4.1 (2018-12-05)
------------------

**Bugfixes**

* Add chardet to `install_requires` to fix [#210](https://github.com/socialcopsdev/camelot/issues/210). More details in [pdfminer.six#213](https://github.com/pdfminer/pdfminer.six/issues/213).

0.4.0 (2018-11-23)
------------------

**Improvements**

* [#102](https://github.com/socialcopsdev/camelot/issues/102) Detect tables automatically when Stream is used. [#206](https://github.com/socialcopsdev/camelot/pull/206) Add implementation of Anssi Nurminen's table detection algorithm by Vinayak Mehta.

0.3.2 (2018-11-04)
------------------

**Improvements**

* [#186](https://github.com/socialcopsdev/camelot/issues/186) Add `_bbox` attribute to table. [#193](https://github.com/socialcopsdev/camelot/pull/193) by Vinayak Mehta.
    * You can use `table._bbox` to get coordinates of the detected table.

0.3.1 (2018-11-02)
------------------

**Improvements**

* Matplotlib is now an optional requirement. [#190](https://github.com/socialcopsdev/camelot/pull/190) by Vinayak Mehta.
    * You can install it using `$ pip install camelot-py[plot]`.
* [#127](https://github.com/socialcopsdev/camelot/issues/127) Add tests for plotting. Coverage is now at 87%! [#179](https://github.com/socialcopsdev/camelot/pull/179) by [Suyash Behera](https://github.com/Suyash458).

0.3.0 (2018-10-28)
------------------

**Improvements**

* [#162](https://github.com/socialcopsdev/camelot/issues/162) Add password keyword argument. [#180](https://github.com/socialcopsdev/camelot/pull/180) by [rbares](https://github.com/rbares).
    * An encrypted PDF can now be decrypted by passing `password='<PASSWORD>'` to `read_pdf` or `--password <PASSWORD>` to the command-line interface. (Limited encryption algorithm support from PyPDF2.)
* [#139](https://github.com/socialcopsdev/camelot/issues/139) Add suppress_warnings keyword argument. [#155](https://github.com/socialcopsdev/camelot/pull/155) by [Jonathan Lloyd](https://github.com/jonathanlloyd).
    * Warnings raised by Camelot can now be suppressed by passing `suppress_warnings=True` to `read_pdf` or `--quiet` to the command-line interface.
* [#154](https://github.com/socialcopsdev/camelot/issues/154) The CLI can now be run using `python -m`. Try `python -m camelot --help`. [#159](https://github.com/socialcopsdev/camelot/pull/159) by [Parth P Panchal](https://github.com/pqrth).
* [#165](https://github.com/socialcopsdev/camelot/issues/165) Rename `table_area` to `table_areas`. [#171](https://github.com/socialcopsdev/camelot/pull/171) by [Parth P Panchal](https://github.com/pqrth).

**Bugfixes**

* Raise error if the ghostscript executable is not on the PATH variable. [#166](https://github.com/socialcopsdev/camelot/pull/166) by Vinayak Mehta.
* Convert filename to lowercase to check for PDF extension. [#169](https://github.com/socialcopsdev/camelot/pull/169) by [Vinicius Mesel](https://github.com/vmesel).

**Files**

* [#114](https://github.com/socialcopsdev/camelot/issues/114) Add Makefile and make codecov run only once. [#132](https://github.com/socialcopsdev/camelot/pull/132) by [Vaibhav Mule](https://github.com/vaibhavmule).
* Add .editorconfig. [#151](https://github.com/socialcopsdev/camelot/pull/151) by [KOLANICH](https://github.com/KOLANICH).
* Downgrade numpy version from 1.15.2 to 1.13.3.
* Add requirements.txt for readthedocs.

**Documentation**

* Add "Using conda" section to installation instructions.
* Add readthedocs badge.

0.2.3 (2018-10-08)
------------------

* Remove hard dependencies on requirements versions.

0.2.2 (2018-10-08)
------------------

**Bugfixes**

* Move opencv-python to extra\_requires. [#134](https://github.com/socialcopsdev/camelot/pull/134) by Vinayak Mehta.

0.2.1 (2018-10-05)
------------------

**Bugfixes**

* [#121](https://github.com/socialcopsdev/camelot/issues/121) Fix ghostscript subprocess call for Windows. [#124](https://github.com/socialcopsdev/camelot/pull/124) by Vinayak Mehta.

**Improvements**

* [#123](https://github.com/socialcopsdev/camelot/issues/123) Make PEP8 compatible. [#125](https://github.com/socialcopsdev/camelot/pull/125) by [Oshawk](https://github.com/Oshawk).
* [#110](https://github.com/socialcopsdev/camelot/issues/110) Add more tests. Coverage is now at 84%!
    * Add tests for `__repr__`. [#128](https://github.com/socialcopsdev/camelot/pull/128) by [Vaibhav Mule](https://github.com/vaibhavmule).
    * Add tests for CLI. [#122](https://github.com/socialcopsdev/camelot/pull/122) by [Vaibhav Mule](https://github.com/vaibhavmule) and [#117](https://github.com/socialcopsdev/camelot/pull/117) by Vinayak Mehta.
    * Add tests for errors/warnings. [#113](https://github.com/socialcopsdev/camelot/pull/113) by Vinayak Mehta.
    * Add tests for output formats and parser kwargs. [#126](https://github.com/socialcopsdev/camelot/pull/126) by Vinayak Mehta.
* Add Python 3.5 and 3.7 support. [#119](https://github.com/socialcopsdev/camelot/pull/119) by Vinayak Mehta.
* Add logging and warnings.

**Documentation**

* Copyedit all documentation. [#112](https://github.com/socialcopsdev/camelot/pull/112) by [Christine Garcia](https://github.com/christinegarcia).
* [#115](https://github.com/socialcopsdev/camelot/issues/115) Update issue labels in contributor's guide. [#116](https://github.com/socialcopsdev/camelot/pull/116) by [Johnny Metz](https://github.com/johnnymetz).
* Update installation instructions for Windows. [#124](https://github.com/socialcopsdev/camelot/pull/124) by Vinayak Mehta.

**Note**: This release also bumps the version for numpy from 1.13.3 to 1.15.2 and adds a MANIFEST.in. Also, openpyxl==2.5.8 is a new requirement and pytest-cov==2.6.0 is a new dev requirement.

0.2.0 (2018-09-28)
------------------

**Improvements**

* [#81](https://github.com/socialcopsdev/camelot/issues/81) Add Python 3.6 support. [#109](https://github.com/socialcopsdev/camelot/pull/109) by Vinayak Mehta.

0.1.2 (2018-09-25)
------------------

**Improvements**

* [#85](https://github.com/socialcopsdev/camelot/issues/85) Add Travis and Codecov.

0.1.1 (2018-09-24)
------------------

**Documentation**

* Add documentation fixes.

0.1.0 (2018-09-24)
------------------

* Rebirth!
