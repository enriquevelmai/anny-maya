# NOTICE

anny-maya is licensed under the Apache License, Version 2.0 (see `LICENSE`).
Copyright 2026 Enrique Velasco Mairal.

This repository ships **code and documentation only**. The pieces below are
**not stored in the repository** — the module downloads them into itself on
first run (`scripts/anny_maya/services/bootstrap.py`), so these notices
describe what gets installed on your machine.

## NAVER Anny (downloaded, unmodified subset; code ported to numpy)

On first run, `external/anny_data/` is populated with a subset of the
[naver/anny](https://github.com/naver/anny) repository at the exact commit
pinned in `external/ANNY_DATA.json` (release v0.6.1,
`d6fc027ced5c17b6b0775dee944096ade7a9ef80`), downloaded by SHA and extracted
unmodified. Only these folders are taken: `src/anny/data/mpfb2/3dobjs`,
`mpfb2/targets`, `mpfb2/rigs/standard/{rig,weights}.default.json`,
`faceunits01`, `cached/anny.pth`, plus Anny's `LICENSE`/`README.md` and
`mpfb2/LICENSE.md`.

- **Anny** — Copyright (C) 2025 NAVER Corp., Apache License 2.0
  (https://github.com/naver/anny/blob/main/LICENSE). Paper: *Anny: a
  parametric human body model for all ages* (arXiv:2511.03589).
  `external/_anny_core.py` is a numpy re-implementation (port) of Anny's
  model construction and evaluation for the `anny` rig and `anny-quads`
  topology (`anny/models/full_model.py`, `model_data.py`,
  `model_transforms.py`, `phenotype.py`, `rigged_model.py`,
  `utils/kinematics.py`, `skinning/skinning.py`, `utils/interpolation.py`);
  it is a derivative work of that Apache-2.0 code. `cached/anny.pth`
  (precomputed rig orientations) is read with a torch-free unpickler.
- **MakeHuman / MPFB2 assets** (`data/mpfb2`: base mesh, macro and local
  targets, rig and skin weights) — by the MakeHuman community, adapted by
  NAVER, licensed **CC0 1.0 Universal**
  (https://github.com/makehumancommunity/mpfb2).
- **Face Units asset pack** (`data/faceunits01`: the 52 ARKit facial
  actions) — by Mika Suominen, **CC0 1.0 Universal**
  (https://static.makehumancommunity.org/assets/assetpacks/index.html).

**Excluded on purpose:** nothing non-commercial is downloaded or used — no
SMPL / SMPL-X topology or assets, no `anny[smpl]` extras, no NAVER
non-commercial zip. Everything this tool installs allows commercial use.

This project is **not affiliated with, sponsored by, or endorsed by NAVER**
or the MakeHuman project. "Anny" and "NAVER" are used only to describe the
model this tool loads (Apache-2.0 grants no trademark rights, §6).

## Google GNM head model (optional; downloaded unmodified when first used)

When the GNM head is first switched on (Head tab), `external/gnm_data/` is
populated with `gnm/shape/data/versions/v3_0/gnm_head.npz`,
`gnm/shape/data/semantic_sampler/expression_decoder_model.h5` plus `LICENSE`
and the READMEs from [google/GNM](https://github.com/google/GNM) at the
exact commit pinned in `external/GNM_DATA.json`
(`9c9419f191edd68644ef5cb6572e238248e9a81c`), unmodified.

- **GNM** ("Generative aNthropometric Model") - Copyright 2026 Google LLC,
  Apache License 2.0 (https://github.com/google/GNM/blob/main/LICENSE).
  `external/_head.py` evaluates the model's bind pose with numpy following
  `gnm/shape/gnm_common.py` (a derivative of that Apache-2.0 code) and
  attaches it to the Anny body. `external/_arkit.py` runs the expression
  decoder's forward pass in numpy (following
  `gnm/shape/semantic_sampler.py`, same licence) to derive the head's ARKit
  shapes, cached as `gnm_data/arkit_compiled_v1.npz` on your machine.

Not affiliated with, sponsored by, or endorsed by Google.

## Modified files (Apache-2.0 section 4)

These files of this repository are derivative works of the Apache-2.0 code
above, re-implemented by Enrique Velasco Mairal; each carries the original
copyright and a notice of the changes in its header:

- `external/_anny_core.py` - numpy port of NAVER Anny's model construction
  and evaluation.
- `external/_head.py` - GNM bind-pose evaluation (after Google GNM's
  `gnm_common.py`) plus the Anny attachment.
- `external/_arkit.py` - GNM expression-decoder forward pass (after Google
  GNM's `semantic_sampler.py`) and the ARKit-named shapes built from it.

"ARKit" is a trademark of Apple Inc.; anny-maya only uses the standard
blendshape names (eyeBlinkLeft, jawOpen, ...) to describe its face shapes and
is not affiliated with Apple.

## Bundled Python runtime (set up on first run)

On first run, `runtime/` is populated with the module's own Python. On
**Windows** this is the official **embeddable CPython 3.11.9**
(https://www.python.org/downloads/), redistributed under the **PSF License
Agreement** (also at `runtime/LICENSE.txt` once downloaded). On
**Linux/macOS** it is a venv created from the machine's own `python3`
(nothing is redistributed). Its only dependency is **NumPy**
(BSD-3-Clause); switching on the GNM head also installs **h5py**
(BSD-3-Clause), used once to read GNM's expression decoder. PyTorch is
never installed.

## In-app license viewer

**Anny ▸ Licenses…** lists this module's license plus every license actually
installed on your machine (Anny, the CC0 assets, GNM, CPython, NumPy,
h5py).
