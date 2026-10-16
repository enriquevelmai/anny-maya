"""First-run bootstrap: set up everything that is fetched/built on demand.

The repo ships only code + docs. Two pieces are produced on demand, both
inside the module folder (nothing touches the system):

  runtime/               the module's own Python 3 with numpy
                         - Windows: portable embeddable CPython 3.11.9
                           (python.org, ~11 MB) + pip via get-pip.py
                         - Linux/macOS: a venv built from the system
                           ``python3`` (or Maya's own mayapy as a fallback)
  external/anny_data/    NAVER's Anny assets (CC0 MakeHuman/MPFB2 targets,
                         Face Units, rig + weights, the anny.pth rig cache),
                         extracted from the naver/anny commit pinned in
                         external/ANNY_DATA.json (exact SHA, never a branch),
                         then compiled once into a fast-loading .npz.

(The posix venv references the host interpreter it was built from - if you
move the module folder afterwards, delete ``runtime/`` and let it rebuild.)
"""

from __future__ import annotations

import glob
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from anny_maya.core import config

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

PY_VER = "3.11.9"
EMBED_URL = ("https://www.python.org/ftp/python/%s/"
             "python-%s-embed-amd64.zip" % (PY_VER, PY_VER))
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# The numpy port needs numpy and nothing else (no PyTorch, ever).
CORE_PACKAGES = ["numpy"]

_PTH = "python311.zip\n.\nLib\\site-packages\n\nimport site\n"

RUNTIME_DIR = os.path.join(config.MODULE_ROOT, "runtime")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0


# --- runtime ----------------------------------------------------------------

def runtime_python():
  if IS_WINDOWS:
    return os.path.join(RUNTIME_DIR, "python.exe")
  for name in ("python3", "python"):
    p = os.path.join(RUNTIME_DIR, "bin", name)
    if os.path.isfile(p):
      return p
  return os.path.join(RUNTIME_DIR, "bin", "python3")


def site_packages_dirs():
  if IS_WINDOWS:
    dirs = [os.path.join(RUNTIME_DIR, "Lib", "site-packages")]
  else:
    dirs = glob.glob(os.path.join(RUNTIME_DIR, "lib", "python3*",
                                  "site-packages"))
  return [d for d in dirs if os.path.isdir(d)]


def runtime_available():
  if not os.path.isfile(runtime_python()):
    return False
  return any(os.path.isdir(os.path.join(d, "numpy"))
             for d in site_packages_dirs())


def _download(url, timeout=300, progress=None):
  logger.info("Downloading %s", url)
  req = urllib.request.Request(url, headers={"User-Agent": "anny-maya"})
  with urllib.request.urlopen(req, timeout=timeout) as r:
    total = int(r.headers.get("Content-Length") or 0)
    chunks, got = [], 0
    while True:
      b = r.read(1 << 20)
      if not b:
        break
      chunks.append(b)
      got += len(b)
      if progress:
        progress(got, total)
    return b"".join(chunks)


def _run(cmd, what):
  r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     universal_newlines=True, creationflags=_NO_WINDOW)
  if r.returncode != 0:
    raise RuntimeError("%s failed:\n%s" % (what, r.stdout[-1500:]))
  return r


def _host_python():
  """A Python 3.9+ that can create the module's venv (posix only)."""
  for name in ("python3", "python"):
    exe = shutil.which(name)
    if not exe:
      continue
    try:
      r = subprocess.run(
          [exe, "-c", "import sys; print(sys.version_info[:2] >= (3, 9))"],
          stdout=subprocess.PIPE, universal_newlines=True, timeout=15)
      if r.stdout.strip() == "True":
        return exe
    except Exception:
      continue
  return sys.executable  # mayapy


def _ensure_runtime_windows(status):
  status("Downloading portable Python %s…" % PY_VER)
  blob = _download(EMBED_URL)
  os.makedirs(RUNTIME_DIR, exist_ok=True)
  with zipfile.ZipFile(io.BytesIO(blob)) as z:
    z.extractall(RUNTIME_DIR)
  with open(os.path.join(RUNTIME_DIR, "python311._pth"), "w") as f:
    f.write(_PTH)
  py = runtime_python()
  status("Installing pip…")
  get_pip = os.path.join(RUNTIME_DIR, "get-pip.py")
  with open(get_pip, "wb") as f:
    f.write(_download(GET_PIP_URL))
  _run([py, get_pip, "--no-warn-script-location"], "get-pip")
  os.remove(get_pip)
  status("Installing numpy…")
  site = os.path.join(RUNTIME_DIR, "Lib", "site-packages")
  _run([py, "-m", "pip", "install", "--target", site, *CORE_PACKAGES],
       "pip install")


def _ensure_runtime_posix(status):
  host = _host_python()
  status("Creating Python environment (venv)…")
  try:
    _run([host, "-m", "venv", RUNTIME_DIR], "venv creation")
  except RuntimeError as e:
    raise RuntimeError(
        "Could not create the module's Python environment with %r.\n"
        "On Linux, install your distro's python3-venv package and retry.\n\n%s"
        % (host, e))
  py = runtime_python()
  status("Installing numpy…")
  _run([py, "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade")
  _run([py, "-m", "pip", "install", *CORE_PACKAGES], "pip install")


def ensure_runtime(status=lambda msg, **kw: None):
  """Build/download the module's own Python runtime if it is missing."""
  if runtime_available():
    return False
  if IS_WINDOWS:
    _ensure_runtime_windows(status)
  else:
    _ensure_runtime_posix(status)
  logger.info("Runtime bootstrapped at %s", RUNTIME_DIR)
  return True


# --- Anny data ----------------------------------------------------------------

def pinned():
  with open(config.PIN_FILE) as f:
    return json.load(f)


def installed_data():
  try:
    with open(os.path.join(config.DATA_ROOT, "ANNY_DATA.json")) as f:
      return json.load(f)
  except Exception:
    return {}


def data_available():
  """Downloaded AND matching the commit this tool version pins."""
  return (os.path.isdir(config.DATA_DIR)
          and installed_data().get("sha") == pinned()["sha"])


def compiled_available():
  return bool(glob.glob(os.path.join(config.DATA_ROOT, "anny_compiled_v*.npz")))


def _wanted(rel, pin):
  if any(rel.startswith(s) for s in pin.get("skip", [])):
    return False
  for e in pin["extract"]:
    if e.endswith("/") and rel.startswith(e):
      return True
    if rel == e:
      return True
  return False


def _download_pinned(pin, dest_root, sentinel, label, status):
  """Download ``pin['zip_url']`` (an exact commit), extract only the pinned
  paths into a temp dir next to ``dest_root``, then swap it in."""
  mb = pin.get("approx_mb", 40)

  def progress(got, total):
    status("Downloading %s (%s) - %.0f / %s MB…"
           % (label, pin["sha"][:8], got / 1e6,
              "%.0f" % (total / 1e6) if total else "~%d" % mb), tick=False)

  status("Downloading %s (~%d MB)…" % (label, mb))
  blob = _download(pin["zip_url"], timeout=600, progress=progress)
  status("Extracting %s…" % label)
  tmp = tempfile.mkdtemp(prefix="_dl_", dir=config.EXTERNAL_DIR)
  try:
    n = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
      for info in z.infolist():
        parts = info.filename.split("/", 1)
        if len(parts) < 2 or info.is_dir():
          continue
        rel = parts[1]
        if ".." in rel.split("/") or rel.startswith("/"):
          continue
        if not _wanted(rel, pin):
          continue
        dst = os.path.join(tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(info) as src, open(dst, "wb") as out:
          shutil.copyfileobj(src, out)
        n += 1
    if not os.path.isfile(os.path.join(tmp, *sentinel.split("/"))):
      raise RuntimeError("Downloaded archive does not look like %s." % label)
    with open(os.path.join(tmp, os.path.basename(pin["_file"])), "w") as f:
      json.dump({k: v for k, v in pin.items() if k != "_file"}, f, indent=2)
    old = dest_root + "_old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(dest_root):
      os.rename(dest_root, old)
    os.rename(tmp, dest_root)
    shutil.rmtree(old, ignore_errors=True)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)
  logger.info("%s %s extracted (%d files) to %s", label, pin["sha"][:8], n,
              dest_root)
  try:  # new data needs a fresh worker
    from anny_maya.core import worker
    worker.shutdown_worker()
  except Exception:
    pass


def ensure_data(status=lambda msg, **kw: None, force=False):
  """Download the pinned naver/anny commit and extract only the assets the
  numpy port reads (never smplx/non-commercial content)."""
  if data_available() and not force:
    return False
  pin = pinned()
  pin["_file"] = config.PIN_FILE
  _download_pinned(pin, config.DATA_ROOT, "src/anny/data/cached/anny.pth",
                   "NAVER Anny %s" % pin.get("tag", ""), status)
  return True


# --- Google GNM head (optional, Phase 2) ---------------------------------------

def gnm_pinned():
  with open(config.GNM_PIN_FILE) as f:
    d = json.load(f)
  d["_file"] = config.GNM_PIN_FILE
  return d


def gnm_available():
  try:
    with open(os.path.join(config.GNM_ROOT, "GNM_DATA.json")) as f:
      inst = json.load(f)
  except Exception:
    return False
  return (inst.get("sha") == gnm_pinned()["sha"]
          and os.path.isfile(config.GNM_NPZ)
          and os.path.isfile(config.GNM_DECODER))


# Only the GNM head needs these (a one-time read of GNM's Keras decoder for
# the head's ARKit shapes); h5py's only dependency is numpy, already there.
GNM_PACKAGES = ["h5py"]


def gnm_runtime_available():
  return any(os.path.isdir(os.path.join(d, "h5py"))
             for d in site_packages_dirs())


def ensure_gnm_runtime(status=lambda msg, **kw: None):
  """Add h5py to the module's runtime (GNM head only)."""
  if gnm_runtime_available():
    return False
  status("Installing h5py…")
  py = runtime_python()
  if IS_WINDOWS:
    site = os.path.join(RUNTIME_DIR, "Lib", "site-packages")
    _run([py, "-m", "pip", "install", "--no-deps", "--target", site,
          *GNM_PACKAGES], "pip install h5py")
  else:
    _run([py, "-m", "pip", "install", *GNM_PACKAGES], "pip install h5py")
  try:  # the running worker cannot see the new package
    from anny_maya.core import worker
    worker.shutdown_worker()
  except Exception:
    pass
  return True


def gnm_ready():
  return gnm_available() and gnm_runtime_available()


def ensure_gnm(status=lambda msg, **kw: None, force=False):
  """Download Google's GNM head model (pinned commit) into the module,
  plus h5py in the runtime."""
  done = False
  if force or not gnm_available():
    _download_pinned(gnm_pinned(), config.GNM_ROOT,
                     "gnm/shape/data/versions/v3_0/gnm_head.npz",
                     "Google GNM head model", status)
    done = True
  return ensure_gnm_runtime(status) or done


def ensure_gnm_with_dialog():
  """Maya-facing: confirm + download GNM behind a progress dialog."""
  from maya import cmds as mc
  if gnm_ready():
    return True
  pin = gnm_pinned()
  if not mc.about(batch=True):
    ans = mc.confirmDialog(
        title="GNM head",
        message="The GNM head needs Google's GNM model (~%d MB download, "
                "Apache-2.0, pinned commit %s) and the h5py package "
                "(~3 MB).\nBoth install inside the module folder. "
                "Continue?" % (pin.get("approx_mb", 40), pin["sha"][:8]),
        button=["Download", "Cancel"], defaultButton="Download",
        cancelButton="Cancel", dismissString="Cancel")
    if ans != "Download":
      return False
  from anny_maya.ui.progress import MayaProgress
  steps = {"n": 0}
  try:
    with MayaProgress("GNM head setup", maximum=5) as prog:
      def status(m, tick=True):
        if tick:
          steps["n"] = min(steps["n"] + 1, 4)
        prog.set(steps["n"], m)
      ensure_gnm(status=status)
    return True
  except Exception as e:
    logger.exception("GNM download failed")
    if not mc.about(batch=True):
      mc.confirmDialog(title="GNM head", icon="critical", button=["OK"],
                       message="GNM download failed:\n%s" % e)
    return False


def ensure_compiled(status=lambda msg, **kw: None):
  """Compile the assets into the fast-loading .npz (one time, ~20 s)."""
  if compiled_available():
    return False
  status("Compiling the Anny model (one time, ~20 s)…")
  _run([runtime_python(), config.SERVER_SCRIPT, "--compile"],
       "Anny model compilation")
  return True


def all_available():
  return runtime_available() and data_available() and compiled_available()


def ensure_all(status=lambda msg, **kw: None):
  done = []
  if ensure_runtime(status):
    done.append("runtime")
  if ensure_data(status):
    done.append("data")
  if ensure_compiled(status):
    done.append("compiled")
  return done


def ensure_all_with_dialog():
  """Maya-facing: confirm + bootstrap behind a progress dialog."""
  from maya import cmds as mc
  missing = []
  if not runtime_available():
    missing.append("portable Python + numpy (~30 MB download)" if IS_WINDOWS
                   else "Python environment (venv from your python3 + numpy)")
  if not data_available():
    pin = pinned()
    missing.append("NAVER Anny %s assets (~%d MB download, CC0/Apache-2.0)"
                   % (pin.get("tag", pin["sha"][:8]), pin.get("approx_mb", 48)))
  if not missing and not compiled_available():
    missing.append("one-time model compilation (~20 s)")
  if not missing:
    return True
  if not mc.about(batch=True):
    ans = mc.confirmDialog(
        title="Anny first-run setup",
        message="Anny needs to set up:\n- %s\n\nEverything installs inside "
                "the module folder. Continue?" % "\n- ".join(missing),
        button=["Download", "Cancel"], defaultButton="Download",
        cancelButton="Cancel", dismissString="Cancel")
    if ans != "Download":
      return False
  from anny_maya.ui.progress import MayaProgress
  steps = {"n": 0}
  try:
    with MayaProgress("Anny first-run setup", maximum=12) as prog:
      def status(m, tick=True):
        if tick:
          logger.info("%s", m)
          steps["n"] = min(steps["n"] + 1, 11)
        prog.set(steps["n"], m)
      ensure_all(status=status)
    ok, msg = True, "Setup complete - Anny is ready."
  except Exception as e:
    logger.exception("bootstrap failed")
    ok, msg = False, "Setup failed:\n%s" % e
  if not mc.about(batch=True):
    mc.confirmDialog(title="Anny first-run setup", message=msg, button=["OK"],
                     icon="information" if ok else "critical")
  return ok
