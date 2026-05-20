# mypyc compilation (planned for v2.x)

This file documents how Camelot intends to use [mypyc](https://mypyc.readthedocs.io)
to ship compiled hot-path modules in pre-built wheels. **Nothing in this
plan is shipped yet** — the `[speedup]` extra in `pyproject.toml` is a
contract placeholder, not a working build. The doc exists so future
contributors don't relitigate the design decisions every time perf work
comes up.

## Why mypyc

The post-1.0 perf pass (#718 / #727 / #731 / #733 / #732) covered the
biggest wins available in pure Python + NumPy: vectorised
`text_in_bbox`, vectorised + bisect-backed `get_table_index`, one
`playa.open` per call instead of two, etc. The remaining Python-level
overhead in the hot loop is mostly Python's own interpreter dispatch
on small integer arithmetic, attribute lookups on textline objects,
and per-call function-call costs. mypyc removes most of that.

Realistic expected wins after compiling `camelot/utils.py` alone:
**2-3× on the stream / network parse paths**, smaller on lattice
(lattice is already cv2-dominated). Measured against the post-NumPy
baseline, not pre-#718.

## Staged plan

mypyc only really helps fully-annotated, mypy-clean modules. Today the
annotation coverage in `camelot/` is mixed (most parsers ~50%, utils
~30%). The plan in order:

1. **Annotation completion on `camelot/utils.py`.** Independent value
   (mypy in CI catches real bugs), and the only module compiled in the
   pilot. ~1-2 days of focused work, no behavioural change.
2. **`setup.py` shim that opts into mypyc via env var.**
   ```python
   # setup.py
   import os
   from setuptools import setup
   ext_modules = []
   if os.environ.get("CAMELOT_MYPYC"):
       from mypyc.build import mypycify
       ext_modules = mypycify(
           ["camelot/utils.py"],
           opt_level="3",
           debug_level="1",
       )
   setup(ext_modules=ext_modules)
   ```
   pyproject.toml stays the source of truth for everything *but* the
   compile step.
3. **`camelot-py[speedup]` extra in `pyproject.toml`** that pulls in
   `mypy` + a working C compiler at install time. Most users won't
   need this — they'll get pre-built wheels from PyPI.
4. **`cibuildwheel` workflow** that builds + signs pre-compiled wheels
   on push to release tags, for the standard manylinux + macos +
   windows × Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14 matrix. Wheels
   carry the compiled `.so` / `.pyd`; the sdist remains pure Python so
   anyone unable to build the compiled extension still works.

## Non-goals

- We are **not** compiling `camelot/core.py`. The `Table` /
  `TableList` classes have dynamic-attribute and pandas-bridge usage
  that doesn't fit mypyc cleanly (mypyc disallows attribute additions
  on compiled classes; `Table.df`, `Table.confidence`, etc. would all
  need explicit type-decl shapes).
- We are **not** compiling the parsers (`camelot/parsers/*.py`).
  They're full of `**kwargs` and runtime-dispatched parser-by-flavor
  composition — mypyc disallows `**kwargs` on compiled functions in
  most configurations. The cost-benefit tilts the wrong way for now.
- We are **not** compiling on import (e.g. via numba `@jit`). mypyc's
  ahead-of-time model is what lets us ship compiled wheels; runtime
  JIT would re-introduce the build dep at every camelot install.

## Constraints inherited

The C compiler requirement at install time (`pip install camelot-py[speedup]`
*without* a pre-built wheel) puts manylinux/wheels users in a different
class from source-installers. The expected steady state:

| install command | result |
|---|---|
| `pip install camelot-py` on a supported platform | pre-built wheel, mypyc-compiled `utils.py`, no build deps needed |
| `pip install camelot-py` on an exotic platform | sdist → pure-Python fallback, no `.so` |
| `pip install camelot-py[speedup]` | sdist + build, requires `mypy` + C toolchain |

This matches the [Black project's](https://github.com/psf/black) mypyc
rollout: pre-built wheels are the happy path, sdist fallback is the
escape valve.

## When this PR will look "done"

When `pip install camelot-py` on Python 3.12 on Linux installs a wheel
whose `camelot/utils.so` exists and whose `text_in_bbox` /
`get_table_index` benchmark is 2-3× faster than the post-#733 pure-
Python version — without the user having to know mypyc exists.
