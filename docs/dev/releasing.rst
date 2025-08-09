Making a New Release
====================

This document outlines the process for creating a new release of `camelot-py`.

The release process is semi-automated using GitHub Actions and `release-drafter`.

Prerequisites
-------------

- You must have maintainer access to the `camelot-dev/camelot` repository.

Release Steps
-------------

1.  **Drafting the Release**

    Every time a pull request is merged into the `master` branch, the `release-drafter` GitHub Action will automatically update a draft release. This draft will include all the changes since the last release. You can view the draft under the "Releases" section of the repository.

2.  **Publishing the Release**

    When you are ready to create a new release, follow these steps:

    a. Navigate to the `Releases <https://github.com/camelot-dev/camelot/releases>`_ page of the repository.

    b. You should see a draft release at the top of the page. Click the "Edit" button (pencil icon) next to the draft release.

    c. Review the release notes that have been automatically generated. You can edit them if needed.

    d. **Crucially, update the version number in the "Tag version" field.** Follow `semantic versioning <https://semver.org/>`_. For example, if the last release was `v1.0.0`, the new one could be `v1.1.0` for a minor release or `v1.0.1` for a patch release.

    e. Once you are satisfied with the release notes and version number, click the "Publish release" button.

3.  **Automated Publishing**

    Once you publish the release, a GitHub Action will automatically be triggered to:

    - Build the Python distribution (wheel and source tarball).
    - Publish the distribution to PyPI.
    - Sign the distribution with Sigstore.
    - Upload the distribution and signatures as assets to the GitHub release.

    You can monitor the progress of this action under the "Actions" tab of the repository.

And that's it! The new release will be available on PyPI and GitHub.
