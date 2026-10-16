"""Drag-and-drop installer for the anny-maya module.

Usage: drag this file from Explorer/Finder into a running Maya viewport. Maya
calls ``onMayaDroppedPythonFile`` below, which COPIES the module into your
Maya ``modules`` folder, registers it, and loads it into the current session:
an "Anny" menu appears immediately, and the module auto-loads on every future
launch.

Because the install is a copy (``~/Documents/maya/modules/anny-maya``), you
can **delete the downloaded zip and the extracted folder afterwards**. Any
already-downloaded runtime / Anny data inside the extracted folder is carried
along so it isn't re-downloaded.

Lives at the top of the repo for easy discovery; the module itself is the
``anny-maya`` subfolder next to this file.
"""

from __future__ import annotations

import fnmatch
import os
import shutil

_MODULE_DIR = "anny-maya"
_MOD_FILE = "ANNY.mod"


def _this_dir():
  try:
    return os.path.dirname(os.path.abspath(__file__))
  except NameError:
    import inspect
    return os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda: 0)))


def _find_module_root(here):
  """The folder containing ``scripts/anny_maya`` (repo-top layout, or the
  installer sitting inside the module)."""
  if os.path.isdir(os.path.join(here, "scripts", "anny_maya")):
    return here
  candidate = os.path.join(here, _MODULE_DIR)
  if os.path.isdir(os.path.join(candidate, "scripts", "anny_maya")):
    return candidate
  return None


def _write_mod(module_root, modules_dir):
  if not os.path.isdir(modules_dir):
    os.makedirs(modules_dir)
  mod_path = os.path.join(modules_dir, _MOD_FILE)
  with open(mod_path, "w") as f:
    f.write("+ ANNY 1.0 %s\n"
            "scripts: scripts\n"
            "PYTHONPATH +:= scripts\n" % module_root.replace("\\", "/"))
  return mod_path


def _copy_module(src_root, dst_root):
  """Copy the module (skipping caches/scratch and dev-only folders).

  On Linux/macOS the runtime is a venv whose internals hardcode its absolute
  path, so it is skipped and rebuilt in place by the first-run setup. (The
  Windows runtime is a relocatable embeddable Python and is carried along.)
  """
  patterns = ["__pycache__", "*.pyc", "_out*", "_sess*", "_fixtures", ".git",
              ".devvenv", "*_old"]
  if os.name != "nt":
    patterns.append("runtime")

  def ignored(name):
    return any(fnmatch.fnmatch(name, p) for p in patterns)

  # Manual walk: shutil.copytree(dirs_exist_ok=...) needs Python 3.8 and
  # Maya 2022 ships 3.7.
  for root, dirs, files in os.walk(src_root):
    dirs[:] = [d for d in dirs if not ignored(d)]
    rel = os.path.relpath(root, src_root)
    out = dst_root if rel == "." else os.path.join(dst_root, rel)
    if not os.path.isdir(out):
      os.makedirs(out)
    for fname in files:
      if not ignored(fname):
        shutil.copy2(os.path.join(root, fname), os.path.join(out, fname))


def onMayaDroppedPythonFile(*args):
  from maya import cmds as mc

  src_root = _find_module_root(_this_dir())
  if src_root is None:
    mc.confirmDialog(
        title="Anny install failed", icon="critical",
        message="Could not find the anny-maya module next to this installer.\n"
                "Keep drag_and_drop_install.py in the extracted repo, beside "
                "the 'anny-maya' folder.", button=["OK"])
    return

  modules_dir = os.path.join(mc.internalVar(userAppDir=True), "modules")
  install_root = os.path.join(modules_dir, _MODULE_DIR)
  already_there = (os.path.normcase(os.path.normpath(src_root))
                   == os.path.normcase(os.path.normpath(install_root)))

  if not already_there:
    if os.path.isdir(install_root):
      ans = mc.confirmDialog(
          title="Anny install",
          message="An anny-maya install already exists at:\n%s\n\nReplace its "
                  "code/docs with this copy? (The downloaded runtime and Anny "
                  "data there are kept.)" % install_root,
          button=["Replace", "Cancel"], defaultButton="Replace",
          cancelButton="Cancel", dismissString="Cancel")
      if ans != "Replace":
        return
    try:
      _copy_module(src_root, install_root)
    except Exception as e:
      mc.confirmDialog(title="Anny install failed", icon="critical",
                       message="Could not copy the module into:\n%s\n\n%s"
                               % (install_root, e), button=["OK"])
      return

  try:
    mod_path = _write_mod(install_root, modules_dir)
  except Exception as e:
    mc.confirmDialog(title="Anny install failed", icon="critical",
                     message="Could not write the module file:\n%s" % e,
                     button=["OK"])
    return

  # Load into the current session without a restart - from the INSTALLED copy.
  import sys
  scripts_dir = os.path.join(install_root, "scripts")
  if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
  load_err = None
  try:
    import importlib
    for name in list(sys.modules):  # an older copy may already be imported
      if name == "anny_maya" or name.startswith("anny_maya."):
        del sys.modules[name]
    import anny_maya  # noqa: F401
    user_setup = importlib.import_module("userSetup")
    importlib.reload(user_setup)
    user_setup.build_menu()
    try:
      anny_maya.add_shelf_button()
    except Exception:
      pass
    # First run: the repo ships code only; the runtime and Anny's data are
    # downloaded on demand into the installed copy.
    try:
      from anny_maya.services import bootstrap
      if not bootstrap.all_available():
        bootstrap.ensure_all_with_dialog()
    except Exception:
      pass  # opening the panel re-offers the setup if still missing
  except Exception as e:
    load_err = str(e)

  msg = "anny-maya installed to:\n%s\n(module file: %s)\n\n" % (install_root,
                                                              mod_path)
  if not already_there:
    msg += ("This is a self-contained copy - you can now DELETE the "
            "downloaded zip and the extracted folder.\n\n")
  if load_err is None:
    msg += ("Loaded into this session - see the 'Anny' menu and the new "
            "'Anny' shelf button. It also auto-loads on future Maya launches.")
  else:
    msg += ("Installed, but live-load failed (%s).\nRestart Maya to finish "
            "loading." % load_err)
  mc.confirmDialog(title="Anny installed", message=msg, button=["OK"])
