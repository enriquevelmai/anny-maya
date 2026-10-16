"""Menu-level actions (Anny menu): each wrapped so failures pop a dialog."""

from __future__ import annotations

import functools
import logging
import os

from maya import cmds as mc

logger = logging.getLogger(__name__)


def _report_errors(title):
  def deco(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
      try:
        return fn(*a, **kw)
      except Exception as e:
        logger.exception(title)
        if not mc.about(batch=True):
          mc.confirmDialog(title="Anny - %s" % title, icon="critical",
                           message="%s failed:\n%s\n\nSee the Script Editor "
                                   "for the full traceback." % (title, e),
                           button=["OK"])
        return None
    return wrapper
  return deco


def _ready():
  from anny_maya.services import bootstrap
  return bootstrap.all_available() or bootstrap.ensure_all_with_dialog()


@_report_errors("Random Body")
def random_body():
  if not _ready():
    return None
  from anny_maya.core.body import create_body
  b = create_body()
  return b.transform


@_report_errors("Average Body")
def average_body():
  if not _ready():
    return None
  from anny_maya.core.body import create_body
  b = create_body(randomize=False)
  return b.transform


def export_fbx_dialog(transform):
  """Ask where to save, then export ``transform`` (a baked rig)."""
  from anny_maya.scene import export_fbx
  from anny_maya.core import settings
  picked = mc.fileDialog2(fileMode=0, caption="Export Anny rig as FBX",
                          fileFilter="FBX (*.fbx)",
                          startingDirectory=settings.exports_dir(),
                          dialogStyle=2) or []
  if not picked:
    return None
  path = picked[0]
  settings.set_exports_dir(os.path.dirname(path))
  path = export_fbx.export_rigged_fbx(transform, path=path)
  mc.confirmDialog(title="Export FBX", message="Exported:\n%s" % path,
                   button=["OK"])
  return path


@_report_errors("Export FBX")
def export_selected_fbx():
  from anny_maya.scene import rig
  sel = rig.find_rigs(selected_only=True)
  if not sel:
    mc.confirmDialog(title="Export FBX", icon="warning",
                     message="Select a baked Anny rig first "
                             "(Anny panel > Animate > Bake Rig).",
                     button=["OK"])
    return None
  return export_fbx_dialog(sel[0])


@_report_errors("Bake Rig")
def bake_selected():
  from anny_maya.core.body import AnnyBody, find_bodies
  from anny_maya.scene import rig
  bodies = find_bodies(selected_only=True)
  if not bodies:
    mc.confirmDialog(title="Bake Rig", icon="warning",
                     message="Select an Anny body first.", button=["OK"])
    return None
  return rig.bake_rig(AnnyBody.adopt(bodies[0]))
