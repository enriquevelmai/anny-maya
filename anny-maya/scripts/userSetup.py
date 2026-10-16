"""Maya startup hook: add an 'Anny' menu to the main window.

Placed on PYTHONPATH by ANNY.mod, so Maya auto-executes it at launch. Uses
executeDeferred so the main window exists before we build the menu, and stays
silent in batch/mayapy (no UI) sessions.
"""

import maya.utils
from maya import cmds as mc

_MENU = "annyMenu"


def _menu_image(name):
  try:
    from anny_maya.ui import icons
    return icons.image_file(name, 22)
  except Exception:
    return ""


def _item(label, command, icon=None, annotation=""):
  kwargs = dict(label=label, parent=_MENU, command=command)
  if annotation:
    kwargs["annotation"] = annotation
  img = _menu_image(icon) if icon else ""
  if img:
    kwargs["image"] = img
  return mc.menuItem(**kwargs)


def build_menu():
  if mc.about(batch=True):
    return
  if not mc.window("MayaWindow", exists=True):
    return
  if mc.menu(_MENU, exists=True):
    mc.deleteUI(_MENU)
  mc.menu(_MENU, label="Anny", parent="MayaWindow", tearOff=True)
  _item("Anny Body Panel...", "import anny_maya; anny_maya.show_ui()",
        icon="anny",
        annotation="Open the Anny Body panel (phenotypes, body, face, pose, "
                   "rig).")
  mc.menuItem(divider=True, parent=_MENU)
  _item("Quick: Random Body",
        "from anny_maya.ui import tools; tools.random_body()", icon="dice",
        annotation="Create a new body with random phenotypes.")
  _item("Quick: Average Body",
        "from anny_maya.ui import tools; tools.average_body()", icon="body",
        annotation="Create a new body at the average (all phenotypes 0.5).")
  _item("Bake Selected Body to Rig",
        "from anny_maya.ui import tools; tools.bake_selected()",
        icon="skeleton",
        annotation="Joints + skinCluster + 52 ARKit blendShapes from the "
                   "selected body.")
  _item("Export Selected Rig (FBX)...",
        "from anny_maya.ui import tools; tools.export_selected_fbx()",
        icon="download", annotation="Export the selected baked rig to FBX.")
  mc.menuItem(divider=True, parent=_MENU)
  _item("Add Shelf Button", "import anny_maya; anny_maya.add_shelf_button()",
        icon="add_box", annotation="Add an Anny button to the current shelf.")
  _item("Check for Updates...", "import anny_maya; anny_maya.check_tool_updates()",
        icon="update", annotation="Check GitHub for a newer anny-maya.")
  _item("Run GUI Test", "import anny_maya; anny_maya.run_gui_test()",
        icon="bug", annotation="Run the in-Maya panel smoke test "
                               "(cleans up after itself).")
  _item("Licenses...", "import anny_maya; anny_maya.show_licenses()",
        icon="description",
        annotation="anny-maya, NAVER Anny, MakeHuman CC0 and runtime licences.")


try:
  maya.utils.executeDeferred(build_menu)
except Exception as e:
  import sys
  sys.stderr.write("Anny menu setup failed: %s\n" % e)
