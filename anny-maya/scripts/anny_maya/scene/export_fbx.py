"""Export a baked Anny rig (mesh + face blendShapes + skeleton) to FBX.

Works on meshes produced by :func:`anny_maya.scene.rig.bake_rig`; the file
carries the skinCluster and the 52 ARKit blendShape targets, so it opens in
game engines / other DCCs with no Anny runtime.
"""

from __future__ import annotations

import logging
import os

from maya import cmds as mc
import maya.mel as mel

from anny_maya.core import settings

logger = logging.getLogger(__name__)


def export_rigged_fbx(transform, path=None):
  """Export ``transform`` (a baked rig) and its skeleton to an FBX file.

  ``path`` defaults to the current scene's folder, or
  ``~/Documents/maya/anny_exports/<transform>.fbx`` for unsaved scenes.
  Returns the absolute path written.
  """
  if not mc.objExists(transform):
    raise ValueError("Object does not exist: %s" % transform)
  mc.loadPlugin("fbxmaya", quiet=True)

  if path is None:
    scene = mc.file(query=True, sceneName=True)
    out_dir = os.path.dirname(scene) if scene else settings.exports_dir()
    path = os.path.join(out_dir, "%s.fbx" % transform.split("|")[-1])
  path = os.path.abspath(path)
  if not os.path.isdir(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))

  joints = skin_joints(transform)
  mc.select([transform] + joints, replace=True)
  mel.eval("FBXResetExport")
  mel.eval("FBXExportSmoothingGroups -v true")
  mel.eval("FBXExportShapes -v true")   # blendshape targets
  mel.eval("FBXExportSkins -v true")    # skinCluster
  mel.eval("FBXExportInputConnections -v false")
  mel.eval("FBXExportEmbeddedTextures -v false")
  mel.eval('FBXExport -f "%s" -s' % path.replace("\\", "/"))
  logger.info("Exported '%s' (+%d joints) -> %s", transform, len(joints), path)
  return path


def skin_joints(transform):
  """Joints influencing ``transform``'s skinCluster(s), via history."""
  joints = []
  for skin in mc.ls(mc.listHistory(transform) or [], type="skinCluster") or []:
    for j in mc.skinCluster(skin, query=True, influence=True) or []:
      full = mc.ls(j, long=True)[0]
      if full not in joints:
        joints.append(full)
  if not joints:
    logger.warning("No skin joints found for '%s'; exporting mesh only.",
                   transform)
  return joints
