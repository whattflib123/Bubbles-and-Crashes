# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a research workspace with one primary asset:
- `Whitehouse et al(2023)Real‐Time Monitoring of Bubbles and Crashes_OBES.pdf`

There is no application source tree yet. If code is added, use this structure:
- `src/` for implementation modules
- `tests/` for automated tests
- `data/` for local input datasets (non-sensitive, small samples only)
- `docs/` for notes, methodology, and derived summaries

Keep filenames descriptive and avoid spaces for new files (example: `bubble_detection_notes.md`).

## Build, Test, and Development Commands
No build system is configured yet. Current useful local commands:
- `ls -la` to inspect repository contents
- `file *.pdf` to validate document file types
- `pdftotext "<paper>.pdf" -` to preview extracted text in terminal (if installed)

If code is introduced, add a project-level task runner (`Makefile` or equivalent) and document commands here (for example: `make test`, `make lint`).

## Coding Style & Naming Conventions
No language formatter/linter is configured yet. For new code:
- Use 4-space indentation
- Prefer `snake_case` for Python files/functions and variables
- Use clear module names aligned with domain intent (example: `src/regime_detector.py`)
- Keep functions small and document non-obvious assumptions

Adopt formatting tools with the first code contribution (recommended: `black` + `ruff` for Python).

## Testing Guidelines
There are no automated tests at this time. When adding code:
- Place tests under `tests/`
- Name test files `test_<module>.py`
- Cover parsing, transformation, and edge-case behavior first
- Run tests locally before opening a PR

Prefer `pytest` and include at least one regression test for each bug fix.

## Commit & Pull Request Guidelines
Git metadata is not present in this directory, so no existing commit convention can be inferred.

Use Conventional Commits moving forward:
- `feat: add bubble indicator parser`
- `fix: handle missing date rows in dataset`

PRs should include:
- A short problem statement and solution summary
- Related issue/ticket reference (if available)
- Before/after evidence for behavior changes (logs, tables, or screenshots)
- Notes on data assumptions and limitations
