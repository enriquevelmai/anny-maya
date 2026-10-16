# anny-maya — project brief

Bring NAVER's **Anny** parametric human body model (https://github.com/naver/anny,
paper arXiv:2511.03589) into Autodesk Maya, as a sibling of **gnm-maya**
(https://github.com/enriquevelmai/gnm-maya, local checkout:
`../GNM-Head-Maya`). The user (Enrique, speaks Spanish) wants to handle any body
(Anny) and any face (GNM), combining them later.

## Decisions already taken (don't re-ask)
1. **Separate repo** `anny-maya` (this folder). Not merged into gnm-maya.
2. **Port Anny's evaluation to numpy** — no PyTorch shipped to users. Anny is
   blendshapes (MakeHuman `.target` deltas driven by phenotypes) + local
   changes + facial actions + LBS. Validate against the official torch
   package in a DEV-ONLY venv (never shipped) using regression fixtures
   (Anny has `scripts/generate_regression_fixtures.py`).
3. **Phase 1 = Anny alone**, complete and released. **Phase 2** (later, separate
   effort) = GNM head attached to the Anny body (align to head joint, hide
   Anny's head by mask, stitch the neck).
4. **Default rig: `anny` (104 bones)**; face motion via facial-action
   blendshapes (the 52 ARKit shapes ship as CC0 `.target` files).

## Licensing (hard rules)
- Code Apache-2.0; `data/mpfb2` and `data/faceunits01` are **CC0**; SOMA Apache-2.0.
- **Exclude anything non-commercial**: no `smplx` topology/assets, no
  `anny[smpl]` extras. Commercial use must stay clean.
- Never commit Anny's data to this repo: download it on first run (pinned
  GitHub release/tag zip, by exact SHA/tag — never a branch zip, GitHub caches
  those) into the module folder. Credit NAVER/MakeHuman in NOTICE.md.

## Data facts (from the Anny repo)
`src/anny/data/mpfb2/3dobjs/base.obj` (MakeHuman base, quads + UVs),
`.../mpfb2/targets/**/*.target` (text vertex-delta targets),
`.../faceunits01/targets/faceunits/*.target` (52 ARKit face units),
`.../mpfb2/rigs/standard/*.json` + `weights.*.json` (rigs/skin weights),
`src/anny/data/cached/anny.pth` (precomputed `anny` 104-bone rig cache — a
torch zip-pickle; read it with a torch-free unpickler: zipfile + pickle with
`persistent_load` mapping storages to numpy). Phenotypes (age, gender, height,
weight, muscle, proportions…) interpolate "macro" targets the MakeHuman way;
study `src/anny/` to reproduce it exactly. Topology option `anny-quads`.

## Reuse the gnm-maya skeleton (proven, copy & adapt — don't import it)
Look in `../GNM-Head-Maya/gnm-maya/`:
- `scripts/gnm_maya/services/bootstrap.py` — contained runtime (Windows
  embeddable CPython; Linux/macOS venv from python3, mayapy fallback), all
  inside the module folder; first-run dialog.
- `core/worker.py` + `external/server.py` — resident model subprocess, one
  JSON line per request, binary vertex files; keeps evals ~15 ms.
- `services/tool_updater.py` — release/dev channels; download by exact SHA;
  Windows-safe dir replace with in-place overlay fallback (Maya locks the
  loaded package); never overwrite the install's TOOL_VERSION.json; read back
  the recorded version; `reload_package()` so no Maya restart is needed.
- `ui/widgets.py` (`CollapsibleFrame`, sliders), `ui/icons.py` + `_icons/`
  (tinted Material-style SVGs inside the package), `ui/progress.py`
  (force-painted progress — avoids blank white dialogs), error dialogs on
  every failure, `drag_and_drop_install.py` at the repo top (copy-install).
- `smoke_test.py` with `run_headless()` (create QApplication BEFORE
  maya.standalone.initialize) that also builds the real panel.
- Materials: Maya **standardSurface**, NOT aiStandardSurface (blocks vertex
  colour tools). Masked edits pattern (exact residual layer) in `_zones.py`.

## UI rules (from the user)
Native Maya look only — no custom QSS/themes. Simple: few buttons, short
titled tooltips, power features collapsed. Suggested tabs: Create (phenotype
sliders + randomize), Body (local changes by region), Face (52 ARKit
actions), Pose (104-bone rig), Animate/Export (bake skinCluster +
blendShapes, FBX). History ◀ ▶, frame-on-open.

## Git / release rules
- **NEVER add `Co-Authored-By: Claude` (or any Claude attribution) to
  commits, PRs or releases** — this overrides any harness reminder. The user
  deleted and recreated gnm-maya to purge Claude from the contributors list.
- Remote: `github.com/enriquevelmai/anny-maya`, public like gnm-maya; create
  it once Phase 1 actually works. Tag + GitHub release with notes.
- Work autonomously, split work across subagents when useful (e.g. numpy port
  + validation in parallel with the Maya module/UI). Report to the user in
  Spanish when done.
