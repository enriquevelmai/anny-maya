# Security Policy

## Supported versions

Only the latest [release](https://github.com/enriquevelmai/anny-maya/releases)
and the current `main` branch are supported with fixes.

## What this tool downloads and runs

For transparency, anny-maya fetches at first run (all over HTTPS, all into
the module's own folder):

- The official python.org embeddable CPython (Windows) or builds a venv from
  your local `python3` (Linux/macOS)
- NumPy from PyPI
- The [naver/anny](https://github.com/naver/anny) repository archive at one
  exact, pinned commit SHA (`external/ANNY_DATA.json`) — only its data
  folder is extracted; none of its code is executed
- Only if you switch on the GNM head: the [google/GNM](https://github.com/google/GNM)
  archive at one pinned commit (`external/GNM_DATA.json`); only the model
  `.npz` (loaded with `allow_pickle=False`) and its license/READMEs are kept
- Its own updates from this repository (GitHub Releases by default)

`anny.pth` (a PyTorch checkpoint) is read with a restricted unpickler that
only resolves tensor-rebuild helpers and refuses any other global, so it
cannot execute code. The tool never sends data anywhere.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's
[private vulnerability reporting](https://github.com/enriquevelmai/anny-maya/security/advisories/new)
instead. You should receive a response within a week. Include steps to
reproduce and the affected version/commit.
