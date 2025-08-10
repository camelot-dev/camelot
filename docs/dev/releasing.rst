Making a New Release
====================

This document outlines the process for creating a new release of `camelot-py`.

The release process is semi-automated, starting with a version number change and concluding with a manual release publication on GitHub. This approach ensures versioning is correct and secure.

Release Steps
-------------

1.  **Create a Version Bump Pull Request**

    To begin a new release, a contributor must create a pull request that increments the version number. The version number must be updated in the `pyproject.toml` file.

    For example, to release version `1.0.2`, you would change the following line in the file:

    .. code-block:: toml

        version = "1.0.1"

    to:

    .. code-block:: toml

        version = "1.0.2"

    The title of the pull request should be descriptive, for example: "Bump version to 1.0.2 for release".

2.  **Merge the Pull Request**

    Once the pull request is reviewed and approved, it can be merged into the `main` branch. This action will trigger the `release-drafter` GitHub Action, which will automatically create a draft release with compiled notes from the merged pull requests.

3.  **Publish the GitHub Release**

    This is the **manual and most critical step**. A repository maintainer must go to the `Releases` page on GitHub and find the newly created draft release.

    -   Review the automatically generated release notes.
    -   Ensure the tag is correct (e.g., `v1.0.2`).
    -   Click the **"Publish release"** button.



4.  **Automated Release Workflow**

    Publishing the release will trigger the main release workflow. This workflow will perform the following steps:
    -   Build the Python distribution (wheel and source tarball) from the tagged commit.
    -   Sign the distributions using Sigstore for enhanced supply chain security.
    -   Publish the signed distributions to PyPI.
    -   Upload the signed distributions and their signatures to the GitHub release assets.

You can monitor the progress of this workflow under the "Actions" tab of the repository.
