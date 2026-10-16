"""anny-maya: NAVER's Anny parametric human body inside Autodesk Maya.

The Anny model (a numpy port, validated against the official PyTorch
package) runs in a self-contained Python runtime inside this module
(``<module>/runtime``); Maya only reads the resulting vertex files and builds
the mesh natively via OpenMaya. Nothing here imports numpy, so it is safe
under mayapy.

Public entry points:
    show_ui()              -> opens the Anny Body panel
    create_body(...)       -> builds a body, returns an AnnyBody controller
    show_licenses()        -> bundled/downloaded licenses viewer
"""

import logging as _logging

__all__ = ["show_ui", "create_body", "show_licenses", "add_shelf_button",
           "check_tool_updates", "run_gui_test", "reload_package", "logger"]

__version__ = "1.0.0"


def _configure_logger():
  """One INFO stream handler so actions surface in Maya's Script Editor."""
  lg = _logging.getLogger("anny_maya")
  if not lg.handlers:
    h = _logging.StreamHandler()
    h.setFormatter(_logging.Formatter("[Anny] %(levelname)s: %(message)s"))
    lg.addHandler(h)
    lg.setLevel(_logging.INFO)
    lg.propagate = False
  return lg


logger = _configure_logger()


def _disk_version():
  """__version__ as written ON DISK (newer than the imported one right
  after a tool update)."""
  import os
  import re
  try:
    with open(os.path.join(os.path.dirname(__file__), "__init__.py")) as f:
      m = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
    return m.group(1) if m else __version__
  except Exception:
    return __version__


def reload_package():
  """Drop every anny_maya submodule (and userSetup) from sys.modules so the
  next import loads the code on disk - used after a tool update instead of
  a Maya restart."""
  import sys
  for name in list(sys.modules):
    if name == "anny_maya" or name.startswith("anny_maya.") \
        or name == "userSetup":
      del sys.modules[name]
  try:
    import userSetup
    userSetup.build_menu()
  except Exception:
    pass


def show_ui():
  """Open the panel (lazy Qt import). Self-heals after an update: reloads
  the package first if the code on disk is newer than the loaded one."""
  if _disk_version() != __version__:
    logger.info("anny-maya %s on disk, %s loaded - reloading package",
                _disk_version(), __version__)
    reload_package()
    import anny_maya as fresh
    return fresh.show_ui()
  from anny_maya.ui import panel as ui
  return ui.show()


def create_body(seed=None, randomize=True, name="anny_body"):
  """Build an Anny body (random unless ``randomize=False``); returns its
  AnnyBody controller (set_phenotype / set_local / set_face / set_rotation,
  randomize_*, pose_preset, get_state / set_state)."""
  from anny_maya.core.body import create_body as _create
  return _create(seed=seed, randomize=randomize, name=name)


def show_licenses():
  from anny_maya.ui import licenses
  return licenses.show()


def add_shelf_button(shelf=None):
  from anny_maya.ui import shelf as _shelf
  return _shelf.add_shelf_button(shelf)


def check_tool_updates():
  """Check GitHub for a newer anny-maya and offer to self-update."""
  from anny_maya.services import tool_updater
  return tool_updater.show_update_dialog()


def run_gui_test():
  """Execute the in-Maya GUI smoke test (opens the panel, cleans up)."""
  import os
  from anny_maya.core import config
  path = os.path.join(config.MODULE_ROOT, "tests", "gui_smoke_test.py")
  with open(path) as f:
    code = f.read()
  namespace = {"__file__": path, "__name__": "__anny_gui_test__"}
  exec(compile(code, path, "exec"), namespace)
  return namespace.get("RESULT")
