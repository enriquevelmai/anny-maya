"""The 'Anny' menu in Maya's main window.

Lives inside the package on purpose: Maya executes every userSetup.py on
the path in ONE shared namespace (__main__), so helpers defined at the top
of userSetup.py collide with other tools' (gnm-maya's userSetup has its own
``_item``/``build_menu``: whichever ran last won, and the Anny items ended
up in the GNM menu). Module-level names here are private to this module.
"""

from __future__ import annotations

from maya import cmds as mc

MENU = "annyMenu"


def _menu_image(name):
  try:
    from anny_maya.ui import icons
    return icons.image_file(name, 22)
  except Exception:
    return ""


def _item(menu, label, command, icon=None, annotation=""):
  kwargs = dict(label=label, parent=menu, command=command)
  if annotation:
    kwargs["annotation"] = annotation
  img = _menu_image(icon) if icon else ""
  if img:
    kwargs["image"] = img
  return mc.menuItem(**kwargs)


def build():
  """(Re)build the Anny menu. No-op in batch/mayapy sessions."""
  if mc.about(batch=True):
    return None
  if not mc.window("MayaWindow", exists=True):
    return None
  if mc.menu(MENU, exists=True):
    mc.deleteUI(MENU)
  menu = mc.menu(MENU, label="Anny", parent="MayaWindow", tearOff=True)
  _item(menu, "Anny Body Panel...", "import anny_maya; anny_maya.show_ui()",
        icon="anny",
        annotation="Open the Anny Body panel (phenotypes, body, face, head, "
                   "pose, rig).")
  mc.menuItem(divider=True, parent=menu)
  _item(menu, "Quick: Random Body",
        "from anny_maya.ui import tools; tools.random_body()", icon="dice",
        annotation="Create a new body with random phenotypes.")
  _item(menu, "Quick: Average Body",
        "from anny_maya.ui import tools; tools.average_body()", icon="body",
        annotation="Create a new body at the average (all phenotypes 0.5).")
  _item(menu, "Bake Selected Body to Rig",
        "from anny_maya.ui import tools; tools.bake_selected()",
        icon="skeleton",
        annotation="Joints + skinCluster + 52 ARKit blendShapes from the "
                   "selected body.")
  _item(menu, "Export Selected Rig (FBX)...",
        "from anny_maya.ui import tools; tools.export_selected_fbx()",
        icon="download", annotation="Export the selected baked rig to FBX.")
  mc.menuItem(divider=True, parent=menu)
  _item(menu, "Add Shelf Button",
        "import anny_maya; anny_maya.add_shelf_button()",
        icon="add_box", annotation="Add an Anny button to the current shelf.")
  _item(menu, "Check for Updates...",
        "import anny_maya; anny_maya.check_tool_updates()",
        icon="update", annotation="Check GitHub for a newer anny-maya.")
  _item(menu, "Run GUI Test", "import anny_maya; anny_maya.run_gui_test()",
        icon="bug", annotation="Run the in-Maya panel smoke test "
                               "(cleans up after itself).")
  _item(menu, "Licenses...", "import anny_maya; anny_maya.show_licenses()",
        icon="description",
        annotation="anny-maya, NAVER Anny, MakeHuman CC0, Google GNM and "
                   "runtime licences.")
  return menu
