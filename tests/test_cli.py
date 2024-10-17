import os
import sys
import warnings
from unittest import mock

import pytest
from click.testing import CliRunner

from camelot.cli import cli
from camelot.utils import TemporaryDirectory
from tests.conftest import skip_on_windows


def test_help_output():
    runner = CliRunner()
    prog_name = runner.get_default_prog_name(cli)
    result = runner.invoke(cli, ["--help"])
    output = result.output

    assert prog_name == "camelot"
    assert result.output.startswith("Usage: %(prog_name)s [OPTIONS] COMMAND" % locals())
    assert all(
        v in result.output
        for v in ["Options:", "--version", "--help", "Commands:", "lattice", "stream"]
    )


@skip_on_windows
def test_cli_lattice(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "foo.pdf")
        outfile = os.path.join(tempdir, "foo.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "lattice", infile]
        )
        assert result.exit_code == 0
        assert "Found 1 tables" in result.output

        result = runner.invoke(cli, ["--format", "csv", "lattice", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        result = runner.invoke(cli, ["--output", outfile, "lattice", infile])
        format_error = "Please specify output file format using --format"
        assert format_error in result.output


def test_cli_stream(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "budget.pdf")
        outfile = os.path.join(tempdir, "budget.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "stream", infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        result = runner.invoke(cli, ["--format", "csv", "stream", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        result = runner.invoke(cli, ["--output", outfile, "stream", infile])
        format_error = "Please specify output file format using --format"
        assert format_error in result.output

        result = runner.invoke(
            cli,
            [
                "--margins",
                "1.5",
                "0.5",
                "0.8",
                "--format",
                "csv",
                "--output",
                outfile,
                "stream",
                infile,
            ],
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        result = runner.invoke(
            cli,
            [
                "--margins",
                "1.5",
                "0.5",
                "--format",
                "csv",
                "--output",
                outfile,
                "stream",
                infile,
            ],
        )
        output_error = "Error: Invalid value for '-M' / '--margins': '--format' is not a valid float."
        assert output_error in result.output


@skip_on_windows
def test_cli_parallel(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "diesel_engines.pdf")
        outfile = os.path.join(tempdir, "diesel_engines.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--parallel",
                "--pages",
                "1,2,3",
                "--format",
                "csv",
                "--output",
                outfile,
                "lattice",
                infile,
            ],
        )
        assert result.exit_code == 0
        assert result.output == "Found 2 tables\n"


def test_cli_hybrid(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "budget.pdf")
        outfile = os.path.join(tempdir, "budget.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "hybrid", infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        result = runner.invoke(cli, ["--format", "csv", "hybrid", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        result = runner.invoke(cli, ["--output", outfile, "hybrid", infile])
        format_error = "Please specify output file format using --format"
        assert format_error in result.output


def test_cli_network(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "budget.pdf")
        outfile = os.path.join(tempdir, "budget.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "network", infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"
        result = runner.invoke(cli, ["--format", "csv", "network", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output
        result = runner.invoke(cli, ["--output", outfile, "network", infile])
        format_error = "Please specify output file format using --format"
        assert format_error in result.output


def test_cli_password(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "health_protected.pdf")
        outfile = os.path.join(tempdir, "health_protected.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--password",
                "userpass",
                "--format",
                "csv",
                "--output",
                outfile,
                "stream",
                infile,
            ],
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        output_error = "File has not been decrypted"
        # no password
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "stream", infile]
        )
        assert output_error in str(result.exception)

        # bad password
        result = runner.invoke(
            cli,
            [
                "--password",
                "wrongpass",
                "--format",
                "csv",
                "--output",
                outfile,
                "stream",
                infile,
            ],
        )
        assert output_error in str(result.exception)


def test_cli_output_format(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "health.pdf")

        runner = CliRunner()

        # json
        outfile = os.path.join(tempdir, "health.json")
        result = runner.invoke(
            cli,
            ["--format", "json", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0, f"Output: {result.output}"

        # excel
        outfile = os.path.join(tempdir, "health.xlsx")
        result = runner.invoke(
            cli,
            ["--format", "excel", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0, f"Output: {result.output}"

        # html
        outfile = os.path.join(tempdir, "health.html")
        result = runner.invoke(
            cli,
            ["--format", "html", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0, f"Output: {result.output}"

        # markdown
        outfile = os.path.join(tempdir, "health.md")
        result = runner.invoke(
            cli,
            ["--format", "markdown", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0, f"Output: {result.output}"

        # zip
        outfile = os.path.join(tempdir, "health.csv")
        result = runner.invoke(
            cli,
            [
                "--zip",
                "--format",
                "csv",
                "--output",
                outfile,
                "stream",
                infile,
            ],
        )
        assert result.exit_code == 0, f"Output: {result.output}"


def test_cli_quiet(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "empty.pdf")
        outfile = os.path.join(tempdir, "empty.csv")
        runner = CliRunner()
        with warnings.catch_warnings():
            # the test should fail if any warning is thrown
            # warnings.simplefilter("error")
            try:
                result = runner.invoke(
                    cli,
                    [
                        "--quiet",
                        "--format",
                        "csv",
                        "--output",
                        outfile,
                        "stream",
                        infile,
                    ],
                )
            except Warning as e:
                warning_text = str(e)
                pytest.fail(f"Unexpected warning: {warning_text}")


def test_cli_lattice_plot_type():
    with TemporaryDirectory() as tempdir:
        outfile = os.path.join(tempdir, "lattice_contour.png")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plot_type",
                "contour",
                "--output",
                outfile,
                "--format",
                "--format",
                "png",
            ],
        )
        assert result.exit_code != 0, f"Output: {result.output}"


def test_import_error():
    with mock.patch.dict(sys.modules, {"matplotlib": None}):
        try:
            from camelot.cli import cli
        except ImportError:
            assert cli._HAS_MPL is False
