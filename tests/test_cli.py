import importlib
import importlib.metadata
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
    assert result.output.startswith(
        "Usage: {prog_name} [OPTIONS] COMMAND".format(**locals())
    )


@pytest.mark.parametrize(
    "expected", ["Options:", "--version", "--help", "Commands:", "lattice", "stream"]
)
def test_help_output_lists(expected):
    result = CliRunner().invoke(cli, ["--help"])
    assert expected in result.output


@skip_on_windows
def test_cli_lattice(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "foo.pdf")
        outfile = os.path.join(tempdir, "foo.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["lattice", "--format", "csv", "--output", outfile, infile]
        )
        assert result.exit_code == 0
        assert "Found 1 tables" in result.output

        result = runner.invoke(cli, ["lattice", "--format", "csv", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        # #639: with --format omitted but --output carrying a known extension
        # (.csv here), the format is inferred — the run succeeds rather than
        # bailing out with "Please specify output file format using --format".
        result = runner.invoke(cli, ["lattice", "--output", outfile, infile])
        assert result.exit_code == 0, result.output
        assert "Found 1 tables" in result.output

        # When the --output extension is unknown, the format error still fires.
        unknown_out = os.path.join(tempdir, "foo")
        result = runner.invoke(cli, ["lattice", "--output", unknown_out, infile])
        assert "Please specify output file format using --format" in result.output


@pytest.fixture
def budget_paths(testdir, tmp_path):
    return os.path.join(testdir, "budget.pdf"), str(tmp_path / "budget.csv")


def test_cli_stream(budget_paths):
    infile, outfile = budget_paths
    runner = CliRunner()
    result = runner.invoke(
        cli, ["stream", "--format", "csv", "--output", outfile, infile]
    )
    assert result.exit_code == 0
    assert result.output == "Found 1 tables\n"


def test_cli_stream_requires_output(budget_paths):
    infile, _outfile = budget_paths
    runner = CliRunner()
    result = runner.invoke(cli, ["stream", "--format", "csv", infile])
    output_error = "Error: Please specify output file path using --output"
    assert output_error in result.output


def test_cli_stream_infers_format_from_output_extension(budget_paths):
    # #639: --format inferred from .csv extension when --output is given.
    infile, outfile = budget_paths
    runner = CliRunner()
    result = runner.invoke(cli, ["stream", "--output", outfile, infile])
    assert result.exit_code == 0, result.output
    assert "Found 1 tables" in result.output


def test_cli_stream_margins(budget_paths):
    infile, outfile = budget_paths
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stream",
            "--margins",
            "1.5",
            "0.5",
            "0.8",
            "--format",
            "csv",
            "--output",
            outfile,
            infile,
        ],
    )
    assert result.exit_code == 0
    assert result.output == "Found 1 tables\n"


def test_cli_stream_margins_need_three_values(budget_paths):
    infile, outfile = budget_paths
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stream",
            "--margins",
            "1.5",
            "0.5",
            "--format",
            "csv",
            "--output",
            outfile,
            infile,
        ],
    )
    output_error = (
        "Error: Invalid value for '-M' / '--margins': '--format' is not a valid float."
    )
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
                "lattice",
                "--parallel",
                "--pages",
                "1,2,3",
                "--format",
                "csv",
                "--output",
                outfile,
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
            cli, ["hybrid", "--format", "csv", "--output", outfile, infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        result = runner.invoke(cli, ["hybrid", "--format", "csv", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output

        # #639: --format inferred from .csv extension when --output is given.
        result = runner.invoke(cli, ["hybrid", "--output", outfile, infile])
        assert result.exit_code == 0, result.output
        assert "Found 1 tables" in result.output


def test_cli_network(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "budget.pdf")
        outfile = os.path.join(tempdir, "budget.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["network", "--format", "csv", "--output", outfile, infile]
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"
        result = runner.invoke(cli, ["network", "--format", "csv", infile])
        output_error = "Error: Please specify output file path using --output"
        assert output_error in result.output
        # #639: --format inferred from .csv extension when --output is given.
        result = runner.invoke(cli, ["network", "--output", outfile, infile])
        assert result.exit_code == 0, result.output
        assert "Found 1 tables" in result.output


def test_cli_password(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "health_protected.pdf")
        outfile = os.path.join(tempdir, "health_protected.csv")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "stream",
                "--password",
                "userpass",
                "--format",
                "csv",
                "--output",
                outfile,
                infile,
            ],
        )
        assert result.exit_code == 0
        assert result.output == "Found 1 tables\n"

        output_error = "File has not been decrypted"
        # no password
        result = runner.invoke(
            cli, ["stream", "--format", "csv", "--output", outfile, infile]
        )
        assert output_error in str(result.exception)

        # bad password
        result = runner.invoke(
            cli,
            [
                "stream",
                "--password",
                "wrongpass",
                "--format",
                "csv",
                "--output",
                outfile,
                infile,
            ],
        )
        assert output_error in str(result.exception)


@pytest.mark.parametrize(
    "fmt,ext",
    [("json", ".json"), ("excel", ".xlsx"), ("html", ".html"), ("markdown", ".md")],
)
def test_cli_output_format(testdir, tmp_path, fmt, ext):
    infile = os.path.join(testdir, "health.pdf")
    outfile = str(tmp_path / f"health{ext}")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["stream", "--format", fmt, "--output", outfile, infile],
    )
    assert result.exit_code == 0, f"Output: {result.output}"


def test_cli_output_format_zip(testdir, tmp_path):
    infile = os.path.join(testdir, "health.pdf")
    outfile = str(tmp_path / "health.csv")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stream",
            "--zip",
            "--format",
            "csv",
            "--output",
            outfile,
            infile,
        ],
    )
    assert result.exit_code == 0, f"Output: {result.output}"


def test_cli_quiet(testdir):
    with TemporaryDirectory() as tempdir:
        infile = os.path.join(testdir, "empty.pdf")
        outfile = os.path.join(tempdir, "empty.csv")
        runner = CliRunner()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # --quiet takes a value (is_flag=False); passed bare, it swallowed
            # "--format" and the command exited with a usage error.
            result = runner.invoke(
                cli,
                [
                    "stream",
                    "--quiet",
                    "true",
                    "--format",
                    "csv",
                    "--output",
                    outfile,
                    infile,
                ],
            )
        assert result.exit_code == 0, result.output
        assert [str(w.message) for w in caught] == []


def test_cli_lattice_plot_type():
    with TemporaryDirectory() as tempdir:
        outfile = os.path.join(tempdir, "lattice_contour.png")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "lattice",
                "--plot_type",
                "contour",
                "--output",
                outfile,
                "--format",
                "png",
            ],
        )
        assert result.exit_code != 0, f"Output: {result.output}"


def test_import_error(monkeypatch):
    import camelot
    import camelot.cli as original_cli

    # camelot.cli is already imported, so load a fresh copy with matplotlib
    # hidden instead of expecting the cached import itself to fail.
    monkeypatch.setattr(camelot, "cli", original_cli)
    with mock.patch.dict(sys.modules, {"matplotlib": None}):
        del sys.modules["camelot.cli"]
        fresh_cli = importlib.import_module("camelot.cli")
    assert fresh_cli._HAS_MPL is False


def test_version_package_not_found():
    with mock.patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        from camelot.cli import get_version

        assert get_version() is None


@pytest.mark.parametrize("flavor", ["lattice", "stream", "network", "hybrid"])
def test_cli_plot_without_matplotlib_raises(flavor, testdir, monkeypatch):
    # When matplotlib isn't available, asking any flavor for a --plot_type
    # must raise ImportError before parsing. Covers the
    # `if not _HAS_MPL: raise ImportError` branch in each flavor command —
    # no existing test reaches it, since matplotlib is installed in CI.
    import camelot.cli

    monkeypatch.setattr(camelot.cli, "_HAS_MPL", False)
    infile = os.path.join(testdir, "foo.pdf")
    runner = CliRunner()
    result = runner.invoke(cli, [flavor, "--plot_type", "text", infile])
    assert result.exit_code != 0
    assert isinstance(result.exception, ImportError)


def test_config_set_config():
    """Config.set_config stores values for the cli group's kwargs handoff (#614).

    Direct unit test — the group accepts no options of its own today, so this
    code path isn't exercised by the subcommand integration tests.
    """
    from camelot.cli import Config

    c = Config()
    c.set_config("flag", True)
    c.set_config("name", "lattice")
    assert c.config == {"flag": True, "name": "lattice"}


def test_cli_group_help_invokes_config():
    """The group callback wires ctx.obj to a Config instance (#614)."""
    from click.testing import CliRunner

    from camelot.cli import cli

    runner = CliRunner()
    # Invoke a real subcommand with --help so we hit the group's body
    # (setting ctx.obj) but stop before requiring positional args.
    result = runner.invoke(cli, ["lattice", "--help"])
    assert result.exit_code == 0
    assert "lattice" in result.output.lower()
