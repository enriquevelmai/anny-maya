"""Create a Maya shelf button that opens the Anny panel."""

from __future__ import annotations

import logging

from maya import cmds as mc
import maya.mel as mel

logger = logging.getLogger(__name__)

_LABEL = "Anny Body"
_COMMAND = "import anny_maya; anny_maya.show_ui()"


def _current_shelf():
  top = mel.eval("$tmp = $gShelfTopLevel")
  return mc.tabLayout(top, query=True, selectTab=True)


def add_shelf_button(shelf=None):
  """Add (or replace) an 'Anny Body' button on the given/active shelf."""
  if mc.about(batch=True):
    return None
  shelf = shelf or _current_shelf()
  for child in (mc.shelfLayout(shelf, query=True, childArray=True) or []):
    try:
      if (mc.shelfButton(child, query=True, exists=True)
          and mc.shelfButton(child, query=True, label=True) == _LABEL):
        mc.deleteUI(child)
    except Exception:
      pass
  image = "commandButton.png"
  try:
    from anny_maya.ui import icons
    png = icons.image_file("anny", 32, color="#dcdcdc")
    if png:
      image = png
  except Exception:
    logger.debug("Anny shelf icon unavailable", exc_info=True)
  btn = mc.shelfButton(parent=shelf, label=_LABEL,
                       annotation="Open the Anny Body panel (NAVER Anny "
                                  "parametric human body)",
                       image=image, imageOverlayLabel="Anny",
                       command=_COMMAND, sourceType="python")
  logger.info("Added Anny shelf button on shelf '%s'", shelf)
  return btn
