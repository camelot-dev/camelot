# Changelog

This file documents notable user-visible changes. The day-to-day automatic
release-notes (grouped by PR) are still produced by
[release-drafter](https://github.com/release-drafter/release-drafter)
on `master` — see the GitHub Releases page. This file is the curated
human-readable summary for _major_ releases.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2.0.0 (planned)

The 2.0 release rolls up a substantial backend migration, the resulting
performance work, and a handful of small but user-visible breaking
changes. **Heads-up if upgrading from 1.0.x**:

### Breaking

- **Dropped Python 3.9** (EOL October 2025). Minimum supported is now
  **Python 3.10**. (#740)
- **`flavor="lattice"` default `line_scale` changed from 40 to 15** to match
  the long-standing implementation (the CLI and `read_pdf` docstring used
  to _say_ 40 but the Lattice parser always defaulted to 15). Tables that
  relied on the documented-but-unimplemented `40` will need
  `read_pdf(..., line_scale=40)` explicitly. (#709)
- **`Table.to_excel` now defaults to `index=False, header=False`** to match
  `Table.to_csv`. Excel exports no longer carry the pandas auto-generated
  row index / column header by default. Opt back in with
  `table.to_excel(path, index=True, header=True)`. (#711)
- **`TableList` constructor materialises its iterable input** to a list
  immediately, so `bool()` and `len()` work on `TableList(generator())`
  inputs. A generator passed in will be exhausted at construction time
  rather than at first access. (#710)
- **`PDFHandler.pages` is a property** (was an attribute). Reads work
  unchanged; the value is now resolved lazily on first access. No callers
  in the wild set it, but if you subclassed and overrode it as an
  attribute, that no longer works. (#732)
- **PDF backend swapped from `pypdf` + `pdfminer.six` to
  [`playa-pdf`](https://pypi.org/project/playa-pdf/)**. The dependency
  install set is smaller, encrypted-PDF handling is more accurate, and
  parser hot paths shed several layers of per-page-temp-PDF dance. Pure
  `import camelot` callers should see no API change.

### Added

- **`TableList.filter(...)`** — post-extraction convenience to drop noise /
  low-quality tables by `min_rows`, `min_columns`, `min_accuracy`,
  `max_whitespace`. Returns a new `TableList` (composable); all thresholds
  default to a no-op so nothing is dropped unless asked.
- **`engine="combined"`** for `flavor="lattice"` (and the lattice half of
  `flavor="hybrid"`): unions the PDF's _native vector_ ruled lines into the
  rasterised OpenCV line masks before contour/joint detection, so tables
  whose rules render faintly (vector strokes, anti-aliasing) are still
  found. Safe by construction — raster always runs, vector lines can only
  add, so output is never worse than `engine="raster"`. `engine="auto"`
  now resolves to `combined` when the page carries vector ruled lines,
  else `raster`. (#763)
- **`engine="vector"`** for `flavor="lattice"`: detects tables purely from
  the PDF's native vector ruled lines, **skipping page rasterisation and
  OpenCV entirely** — the fastest path for PDFs whose tables are drawn
  with real vector strokes. (#763)
- **`flavor="auto"`**: render the first requested page, count ruled
  horizontal/vertical lines, pick `lattice` when ruled and `network`
  otherwise. Emits a `UserWarning` naming the chosen flavor. (#737)
- **`Table.confidence`** — unified per-table quality score in `[0, 1]`
  computed as `(accuracy / 100) * (1 - whitespace / 100)`. Now appears as
  a `"confidence"` key in `Table.parsing_report` alongside the existing
  `accuracy` / `whitespace` / `page` / `order`. Suitable for production
  filtering. The whole `parsing_report` schema is now documented in the
  property docstring. (#739)
- **`per_page` parameter** on `read_pdf(..., per_page={...})` — apply
  per-page kwarg overrides (including `flavor`) on top of the global
  kwargs. Useful for multi-layout PDFs where some pages need different
  `table_areas` / `columns` / `flavor` than the rest. Concept originally
  proposed by @sverma25 in #41. (#41)
- **`strip_text=` now accepts a list/tuple of substrings** alongside the
  long-standing per-character `str` form. `strip_text=["[1]", "[2]"]`
  strips those footnote markers as whole substrings;
  `strip_text="[]"` keeps the existing per-character behaviour. (#484)
- **`replace_text` parameter** on `read_pdf` — dict of substring →
  replacement applied to every cell's text just before assignment.
  Unlike `strip_text` (which can only remove), `replace_text` rewrites
  with arbitrary text — useful for collapsing soft-broken words
  (`{" \n": " "}`), normalising abbreviations, or rewriting unit
  names. Keys are matched as literal substrings; when several keys
  could match at the same position the longest one wins. (#482)
- **`read_pdf` accepts `bytes` and binary file-like objects** as
  `filepath`, in addition to str/Path and URLs. `io.BytesIO`, an open
  `"rb"` handle, `requests` response `.raw`, etc. all work. The bytes
  are spilled to a temp file once (so the Lattice OpenCV image
  conversion keeps working) and cleaned up on context-manager exit.
  Long-standing requests #170, #245. (#270)
- **`cpu_count` parameter** on `read_pdf(..., parallel=True, cpu_count=N)`
  and `PDFHandler.parse(...)` — caps the worker count when running in
  parallel. Defaults to all cores; clamped to
  `[1, multiprocessing.cpu_count()]`. (#712)
- **`camelot-py` CLI alias** matching the PyPI package name —
  `uvx camelot-py …` works directly without the `--from camelot-py`
  prefix. (#738)
- **`--format` is now optional** in the CLI: when omitted, the format is
  inferred from the `--output` extension (`.csv`, `.xlsx`, `.html`,
  `.json`, `.md`, `.sqlite`, etc.). (#738)
- **`Table.to_excel` defaults to `index=False, header=False`** (under
  Breaking but worth calling out under Added too — most users will
  prefer the new shape).
- **Python 3.14 stable + 3.15 experimental** rows added to the CI matrix.
  Wheels for both Pythons install correctly on Linux/macOS/Windows. (#706)

### Changed (performance)

- **`text_in_bbox` ≈ 30× faster on busy lattice pages.** The original
  O(n³) duplicate-discard pass became O(n²) in #718, then the whole
  function was NumPy-vectorised in #731 — a 3-4× win on top of #718 on
  realistic 50-500-text-line bboxes. Memory-safe fallback at n > 1500.
- **`get_table_index` 3-13×.** #727 collapsed the row scan + best-overlap
  tracking, #733 added a lazy NumPy + `bisect` row-band lookup
  (`O(log rows)`) plus per-table caches on `Table` (`_rows_np`,
  `_cols_np`, `_rows_disjoint`).
- **`read_pdf` opens the PDF once per call instead of twice.** Page
  resolution is deferred until the parse already has the `playa` handle
  open. Doubles throughput on workloads that loop over many short PDFs.
  (#732)
- `random_string` 4× (#718) and `compute_whitespace` cleanup (#727) —
  small, mostly readability.
- A `bench/` directory now ships a couple of standalone microbenchmarks
  (`bench_get_table_index.py`) and a negative-result bench
  (`bench_negative_results.py`) documenting cases where NumPy did _not_
  help — useful regression net against well-meaning future rewrites.

### Fixed

- **`flavor="auto"` was silently broken** — `_detect_flavor` passed a
  non-existent `resolution=` kwarg to the image backend, so the `TypeError`
  was swallowed and *every* PDF fell back to `network` (never `lattice`).
  Fixed; `auto` now also detects the flavor **per page** and routes ruled
  pages through `engine="combined"`, so mixed cover-page/table documents
  parse correctly. (#763)

- **Windows `PermissionError` when parsing multiple PDFs.** The URL-
  downloaded temp file is now removed on `PDFHandler.__exit__` /
  `close()`; the `os.remove` is wrapped in `try/except OSError` so the
  shutdown path keeps working even when pdfium/playa still holds a
  handle to the file. (#735, closes #537 / #678)
- **`PdfiumBackend` leaks document + image handles.** `convert()` now
  uses `try/finally` so a render that raises still releases pypdfium's
  resources. (#716, closes #660)
- **`TableList(generator)`** no longer raises `TypeError` on `bool()` or
  `len()`. (#710, closes #655)
- **CLI / docs / Lattice default `line_scale` are consistent at 15**
  (see Breaking). (#709, closes #657)
- **`Table.to_excel` no longer emits the meaningless integer-index row
  and integer-header column**. (#711, closes #634)
- **CLI options are position-independent** (they can sit before _or_
  after the file argument on any subcommand). (#614, closes #587)
- **Documentation no longer references `pdfminer`/`pypdf` as the
  backend**; the `playa-pdf` migration is reflected throughout. (#719)
- **opencv-python conflict warning** added to install docs — pip happily
  installs `opencv-python` alongside `opencv-python-headless`, breaking
  `import cv2` at runtime. (#736, closes #645)
- **`how-it-works.rst` Network section** no longer refers to a missing
  plot. (#736, closes #577)

### Security

- **`pypdf<6` (CVE-2025-55197)** is no longer a dependency; replaced by
  `playa-pdf`. The pypdf vulnerability does not apply to current Camelot
  even though Camelot never directly called the affected APIs. (closes
  #643)
- **`PDFTextExtractionNotAllowed` is now actually enforced** for
  encrypted PDFs whose user-password permissions forbid extraction —
  the previous architecture (split-into-per-page-temp-PDFs via pypdf)
  silently dropped the encryption metadata after decryption, so the
  check was effectively a no-op. The playa-based parse path keeps the
  document handle open with permissions intact. Note: for _unencrypted_
  PDFs that claim "no extraction" via `/Perms`, no mechanism in the PDF
  spec actually enforces the flag and Camelot extracts. (closes #590)

### Deprecated / Removed

- The internal `_save_page` per-page-temp-PDF helper is gone — no
  external callers known. (gh-#21, gh-#11)
- `pdfminer.six` is no longer a direct dependency — `playa.miner`
  exposes a PDFMiner-compatible layout API; users who imported through
  Camelot keep working without code changes. (gh-#172)

## 1.0.x

For changes prior to 2.0, see the GitHub Releases page — those were
drafted by `release-drafter` from the merged PR titles.
