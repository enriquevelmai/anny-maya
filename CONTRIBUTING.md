# Contributing to anny-maya

Thanks for your interest! Bug reports, feature ideas and pull requests are
all welcome.

## Reporting bugs

Open an issue using the **Bug report** template. The most useful reports
include:

- Your OS and Maya version (e.g. Windows 11 / Maya 2026)
- The anny-maya version (`Anny ▸ Check for Updates` shows it) or the commit
  id if you run the dev channel
- The full error text from Maya's **Script Editor** (every anny-maya failure
  also pops an error dialog — paste what it says)

## Suggesting features

Open an issue with the **Feature request** template. Screenshots or links to
reference behavior help a lot.

## Pull requests

1. Fork and branch from `main`.
2. Keep the layering: `core/` (state, worker, no Qt), `scene/` (Maya scene
   building), `ui/` (Qt panel), `services/` (bootstrap/updates). Model code
   that needs numpy lives in `external/` and runs in the module's own
   runtime, never inside Maya's Python.
3. Follow the existing style: 2-space indent, docstrings that explain *why*,
   a short titled tooltip for every new UI control, errors surfaced via
   dialogs, native Maya look (no custom stylesheets).
4. Test before pushing:
   - Headless: `mayapy -c "import sys; sys.path.insert(0, r'<repo>/anny-maya/scripts'); from anny_maya import smoke_test; smoke_test.run_headless()"`
   - In Maya: **Anny ▸ Run GUI Test** (cleans up after itself).
   - If you touch `external/_anny_core.py`: the numpy port must still match
     the official package — see *Validation* in the README
     (`anny-maya/tests/validate_vs_official.py`, dev-only venv).
5. Update the README for user-facing changes.

## Project scope

The tool brings [NAVER Anny](https://github.com/naver/anny) (Apache-2.0
code, CC0 MakeHuman assets) to Autodesk Maya. Anny's data, the Python
runtime and validation fixtures are downloaded/generated locally and are
never committed — keep it that way in PRs. Nothing non-commercial (SMPL /
SMPL-X) may be added.
