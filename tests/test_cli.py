# -*- coding: utf-8 -*-

import os

from click.testing import CliRunner

from camelot.cli import cli
from camelot.utils import TemporaryDirectory


testdir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(testdir, "files")


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


def test_cli_lattice():
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "foo.pdf")
        outfile = os.path.join(tempdir, "foo.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "lattice", infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        result = runner.invoke(cli, ["--format", "csv", "lattice", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        result = runner.invoke(cli, ["--output", outfile, "lattice", infile])
        format_error = "Please specify output file format using --format"
        assert format_error in result.output


def test_cli_stream():
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


def test_cli_password():
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

        output_error = "file has not been decrypted"
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


def test_cli_output_format():
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "health.pdf")

        runner = CliRunner()

        # json
        outfile = os.path.join(tempdir, "health.json")
        result = runner.invoke(
            cli,
            ["--format", "json", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0

        # excel
        outfile = os.path.join(tempdir, "health.xlsx")
        result = runner.invoke(
            cli,
            ["--format", "excel", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0

        # html
        outfile = os.path.join(tempdir, "health.html")
        result = runner.invoke(
            cli,
            ["--format", "html", "--output", outfile, "stream", infile],
        )
        assert result.exit_code == 0

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
        assert result.exit_code == 0


def test_cli_quiet():
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "blank.pdf")
        outfile = os.path.join(tempdir, "blank.csv")
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--format", "csv", "--output", outfile, "stream", infile]
        )
        assert "No tables found on page-1" in result.output

        result = runner.invoke(
            cli, ["--quiet", "--format", "csv", "--output", outfile, "stream", infile]
        )
        assert "No tables found on page-1" not in result.output
