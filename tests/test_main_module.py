import runpy

import camelot.cli


def test_main_invokes_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(camelot.cli, "cli", lambda *a, **k: calls.append(True))

    from camelot.__main__ import main

    main()

    assert calls == [True]


def test_run_as_module_invokes_cli(monkeypatch):
    # Executes camelot/__main__.py with __name__ == "__main__" in-process
    # (the `python -m camelot` path), so the module-level `main()` call under
    # the `if __name__ == "__main__"` guard is exercised and measured.
    calls = []
    monkeypatch.setattr(camelot.cli, "cli", lambda *a, **k: calls.append(True))

    runpy.run_module("camelot.__main__", run_name="__main__")

    assert calls == [True]
