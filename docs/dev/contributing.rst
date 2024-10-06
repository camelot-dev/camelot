.. _contributing:


Contributor's Guide
===================

If you're reading this, you're probably looking to contributing to pypdf_table_extraction. *Time is the only real currency*, and the fact that you're considering spending some here is *very* generous of you. Thank you very much!

This document will help you get started with contributing documentation, code, testing and filing issues.

-  This Documentation
-  `Source Code <https://github.com/py-pdf/pypdf_table_extraction>`__
-  `Issue
   Tracker <https://github.com/py-pdf/pypdf_table_extraction/issues>`__
-  :doc:`Code Of Conduct <codeofconduct>`.

Code Of Conduct
---------------

The following quote sums up the **Code Of Conduct**.

    **Be cordial or be on your way**. *--Kenneth Reitz*

Kenneth Reitz has also written an `essay`_ on this topic, which you should read.

.. _essay: https://kennethreitz.org/essays/2013/01/27/be-cordial-or-be-on-your-way

For more info read our full :doc:`Code Of Conduct <codeofconduct>`.

How to report a bug
-------------------

Report bugs on the `Issue
Tracker <https://github.com/py-pdf/pypdf_table_extraction/issues>`__.

When filing an issue, make sure to answer these questions:

-  What did you do?
-  What did you expect to see?
-  What did you see instead?
-  A link to the PDF document that you were trying to extract tables from.
-  The complete traceback.
-  Which operating system and Python version are you using?
-  Which version of this project are you using?
-  Which version of the dependencies are you using?

You can use the following code snippet to find this information::

    import platform; print(platform.platform())
    import sys; print('Python', sys.version)
    import numpy; print('NumPy', numpy.__version__)
    import cv2; print('OpenCV', cv2.__version__)
    import camelot; print('Camelot', camelot.__version__)

- Steps to reproduce the bug, using code snippets. See `Creating and highlighting code blocks`_.

.. _Creating and highlighting code blocks: https://help.github.com/articles/creating-and-highlighting-code-blocks/


Questions
^^^^^^^^^

Please don't use GitHub issues for support questions. A better place for them would be `Stack Overflow`_. Make sure you tag them using the ``pypdf_table_extraction`` tag.

.. _Stack Overflow: http://stackoverflow.com


How to request a feature
------------------------

Request features on the `Issue
Tracker <https://github.com/py-pdf/pypdf_table_extraction/issues>`__.

Your first contribution
-----------------------

A great way to start contributing to pypdf_table_extraction is to pick an issue tagged with the `help wanted`_ or the `good first issue`_ tags. If you're unable to find a good first issue, feel free to contact the maintainer.

.. _help wanted: https://github.com/py-pdf/pypdf_table_extraction/labels/help%20wanted
.. _good first issue: https://github.com/py-pdf/pypdf_table_extraction/labels/good%20first%20issue

Setting up a development environment
------------------------------------

You need Python 3.8+ and the following tools:

-  `Poetry <https://python-poetry.org/>`__
-  `Nox <https://nox.thea.codes/>`__
-  `nox-poetry <https://nox-poetry.readthedocs.io/>`__

Install the package with development requirements:

.. code-block:: console

   $ poetry install

You can now run an interactive Python session, or the command-line
interface:

.. code-block:: console

   $ poetry run python
   $ poetry run pypdf-table-extraction

How to test the project
-----------------------

Run the full test suite:

.. code-block:: console

   $ nox

List the available Nox sessions:

.. code-block:: console

   $ nox --list-sessions

You can also run a specific Nox session. For example, invoke the unit
test suite like this:

.. code-block:: console

   $ nox --session=tests

Unit tests are located in the *tests* directory, and are written using
the `pytest <https://pytest.readthedocs.io/>`__ testing framework.


Pull Requests
-------------

Submit a pull request
^^^^^^^^^^^^^^^^^^^^^

The preferred workflow for contributing to pypdf_table_extraction is to fork the `project repository`_ on GitHub, clone, develop on a branch and then finally submit a pull request. Here are the steps:

.. _project repository: https://github.com/py-pdf/pypdf_table_extraction/


1. Fork the project repository. Click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub.

2. Clone your fork of pypdf_table_extraction from your GitHub account

.. code-block:: console

   $ git clone https://www.github.com/[username]/pypdf_table_extraction

3. Create a branch to hold your changes

.. code-block:: console

    $ git checkout -b my-feature

Always branch out from ``main`` to work on your contribution. It's good practice to never work on the ``main`` branch!

.. note:: ``git stash`` is a great way to save the work that you haven't committed yet, to move between branches.

4. Work on your contribution. Add changed files using ``git add`` and then ``git commit`` them

.. code-block:: console

    $ git add modified_files
    $ git commit

5. Finally, push them to your GitHub fork

.. code-block:: console

    $ git push -u origin my-feature

Now it's time to go to the your fork of pypdf_table_extraction and create a `pull
request <https://github.com/py-pdf/pypdf_table_extraction/pulls>`__! You can `follow these instructions`_ to do the same.

.. _follow these instructions: https://help.github.com/articles/creating-a-pull-request-from-a-fork/

Work on your pull request
^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend that your pull request complies with the following guidelines:

- Make sure your code follows `pep8`_.

.. _pep8: http://pep8.org


- In case your pull request contains function docstrings, make sure you follow the `numpydoc`_ format. All function docstrings in pypdf_table_extraction follow this format. Following the format will make sure that the API documentation is generated flawlessly.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html



- Please create a draft pull request if it is a work in progress. An incomplete pull request's title could be prefixed with [WIP] (to indicate a work in progress). Change the status of your pull request if the contribution is complete and ready for a detailed review. A good `task list`_ in the PR description will ensure that other people get a fair idea of what it proposes to do, which will also increase collaboration.

.. _task list: https://blog.github.com/2013-01-09-task-lists-in-gfm-issues-pulls-comments/

- If contributing new functionality, make sure that you add a unit test for it, while making sure that all previous tests pass.


.. note:: It is recommended to open an issue before starting work on anything. This will allow a chance to talk it over with the contributors and validate your approach.

To run linting and code formatting checks before committing your change,
you can install pre-commit as a Git hook by running the following
command:

.. code-block:: console

   $ nox --session=pre-commit -- install

Your pull request needs to meet the following guidelines for acceptance:

-  The Nox test suite must pass without errors and warnings.
-  Include unit tests. This project maintains 100% code coverage.
-  If your changes add functionality, update the documentation
   accordingly.

Writing Documentation
---------------------

Writing documentation, function docstrings, examples and tutorials is a great way to start contributing to open-source software! The documentation is present inside the ``docs/`` directory of the source code repository.

The documentation is written in `reStructuredText`_, with `Sphinx`_ used to generate these lovely HTML files that you're currently reading (unless you're reading this on GitHub). You can edit the documentation using any text editor and then generate the HTML output by running `make html` in the ``docs/`` directory.

The function docstrings are written using the `numpydoc`_ extension for Sphinx. Make sure you check out how its format guidelines before you start writing one.

.. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html


How to make a release
---------------------

.. note:: *You need to be a project maintainer to make a release.*

Before making a release, go through the following checklist:

- All pull requests for the release have been merged.
- The default branch passes all checks.

Releases are made by publishing a GitHub Release.
A draft release is being maintained based on merged pull requests.
To publish the release, follow these steps:

1. Click **Edit** next to the draft release.
2. Enter a tag with the new version.
3. Enter the release title, also the new version.
4. Edit the release description, if required.
5. Click **Publish Release**.

After publishing the release, the following automated steps are triggered:

- The Git tag is applied to the repository.
- [Read the Docs] builds a new stable version of the documentation.
