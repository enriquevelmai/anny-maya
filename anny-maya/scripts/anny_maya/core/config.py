"""Filesystem layout resolution for the anny-maya module.

The module root is the folder that contains ``scripts/``, ``runtime/`` and
``external/``. We derive it from this file's location so the module works no
matter where the user drops it, as long as the internal layout is preserved.
"""

from __future__ import annotations

import os

# .../<module>/scripts/anny_maya/core/config.py  ->  <module>
_HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

EXTERNAL_DIR = os.path.join(MODULE_ROOT, "external")
SERVER_SCRIPT = os.path.join(EXTERNAL_DIR, "server.py")
# Anny's assets (downloaded on first run from the pinned naver/anny commit).
DATA_ROOT = os.path.join(EXTERNAL_DIR, "anny_data")
DATA_DIR = os.path.join(DATA_ROOT, "src", "anny", "data")
PIN_FILE = os.path.join(EXTERNAL_DIR, "ANNY_DATA.json")
# Google GNM head (optional, downloaded when the GNM head is first enabled).
GNM_ROOT = os.path.join(EXTERNAL_DIR, "gnm_data")
GNM_PIN_FILE = os.path.join(EXTERNAL_DIR, "GNM_DATA.json")
GNM_NPZ = os.path.join(GNM_ROOT, "gnm", "shape", "data", "versions", "v3_0",
                       "gnm_head.npz")
# GNM's semantic expression decoder: source of the head's 52 ARKit shapes.
GNM_DECODER = os.path.join(GNM_ROOT, "gnm", "shape", "data",
                           "semantic_sampler", "expression_decoder_model.h5")


def runtime_python() -> str:
  """Absolute path to the module's bundled Python interpreter.

  Prefers the self-contained ``runtime/`` (Windows: portable embeddable
  Python; Linux/macOS: a venv built on first run). Falls back to a
  locally-built ``venv/`` if present.
  """
  candidates = [
      os.path.join(MODULE_ROOT, "runtime", "python.exe"),          # win portable
      os.path.join(MODULE_ROOT, "runtime", "bin", "python3"),      # posix venv
      os.path.join(MODULE_ROOT, "runtime", "bin", "python"),
      os.path.join(MODULE_ROOT, "venv", "Scripts", "python.exe"),  # dev venv
      os.path.join(MODULE_ROOT, "venv", "bin", "python"),
  ]
  for c in candidates:
    if os.path.isfile(c):
      return c
  raise RuntimeError(
      "Anny's bundled Python not found. Expected one of:\n  "
      + "\n  ".join(candidates)
      + "\nOpen the Anny panel to run the first-run setup again.")


def check_install() -> None:
  """Raise a clear error if a required piece of the module is missing."""
  if not os.path.isfile(SERVER_SCRIPT):
    raise RuntimeError("Missing worker script: %s" % SERVER_SCRIPT)
  if not os.path.isdir(DATA_DIR):
    raise RuntimeError("Anny data not downloaded yet (%s). Open the Anny "
                       "panel to run the first-run setup." % DATA_DIR)
  runtime_python()  # raises with guidance if absent
