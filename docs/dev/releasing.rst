Making a New Release
====================

This document outlines the process for creating a new release of `camelot-py`.

The release process is fully automated using GitHub Actions. It is triggered by a version number change in the `pyproject.toml` file.

Release Steps
-------------

1.  **Create a Version Bump Pull Request**

    To begin a new release, a contributor must create a pull request that increments the version number. The version number must be updated in one place:
    *   `pyproject.toml`

    For example, to release version `1.0.2`, you would change the following line in `pyproject.toml`:

    .. code-block:: toml

        version = "1.0.1"

    to:

    .. code-block:: toml

        version = "1.0.2"

    The title of the pull request should be descriptive, for example: "Bump version to 1.0.2 for release".

2.  **Merge the Pull Request**

    Once the pull request is reviewed and approved, it can be merged into the `main` branch.

3.  **Automated Release**

    When the version bump PR is merged into `main`, a GitHub Actions workflow will automatically perform the following steps:
    *   Detect that the version in `pyproject.toml` has changed.
    *   Create a new git tag corresponding to the new version (e.g., `v1.0.2`).
    *   Build the Python distribution (wheel and source tarball).
    *   Publish the distribution to PyPI.
    *   Use `release-drafter` to create a new GitHub Release with automatically generated release notes for the new tag.

You can monitor the progress of this workflow under the "Actions" tab of the repository.
