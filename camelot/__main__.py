"""Initialize pypdf_table_extraction, formerly known as Camelot."""

__all__ = ("main",)


def main():
    from camelot.cli import cli

    cli()


if __name__ == "__main__":
    main()
