"""Bake a self-sufficient, keyframable Maya rig from an Anny body.

Produces a new mesh with:
  - a blendShape node with the 52 ARKit facial actions (Anny's Face Units,
    exact per-vertex deltas) - native, keyframable, mocap-compatible names;
    with a GNM head: the same 52 ARKit names built on GNM (external/_arkit),
    the GNM expression (+ optional GNM expression modes)
  - the 104-bone ``anny`` skeleton as Maya joints (rest orientations from
    Anny's cached Procrustes frames), skinned with Anny's own weights
    (classic linear skinning = Anny's LBS, so poses match the panel exactly)
  - the panel's current pose and face values applied

The body shape (phenotypes + local changes) is baked into the bind mesh and
skeleton, as in Anny itself (the skeleton depends on the shape). The result
needs no Anny runtime and exports to FBX.
"""

from __future__ import annotations

import json
import logging
import os

import maya.api.OpenMaya as om2
from maya import cmds as mc

from anny_maya.core import meshio
from anny_maya.scene import build as _build

logger = logging.getLogger(__name__)

RIG_ATTR = "annyRig"


def joint_name(label):
  """Maya-safe joint name for an Anny bone label (upperarm01.L ->
  upperarm01_L, toe1-1.L -> toe1_1_L)."""
  return label.replace(".", "_").replace("-", "_")


def find_rigs(selected_only=False):
  nodes = (mc.ls(selection=True, long=False) if selected_only
           else mc.ls(type="transform", long=False))
  return [n for n in (nodes or [])
          if mc.attributeQuery(RIG_ATTR, node=n, exists=True)]


def _read(session, fname, typecode="f"):
  return meshio.read_array(os.path.join(session, fname), typecode)


class _Node(object):
  """A DAG node tracked by UUID, so renames/reparenting and duplicate short
  names (several baked rigs in one scene) never break the lookup."""

  def __init__(self, name):
    self.uuid = mc.ls(name, uuid=True)[0]

  @property
  def path(self):
    return mc.ls(self.uuid, long=True)[0]


def bake_rig(body, name=None, face=True, apply_pose=True, modes=0):
  """Bake ``body`` (an AnnyBody) into a rigged mesh. Returns the transform.

  ``face`` adds the 52 ARKit targets (Anny's, or GNM's with a GNM head);
  with a GNM head also the panel's GNM expression as one target
  (gnmExpression) and ``modes`` the first N GNM expression modes of each
  region."""
  from anny_maya.ui.progress import MayaProgress

  worker = body.worker
  topo = body.topology
  state = body.get_state()
  meta = worker.bake(state, modes=modes if (face and state.get("head")) else 0)
  sess = worker.session_dir
  name = _build.unique_name(name or (body.transform.split("|")[-1] + "_rig"))
  targets = meta["targets"] if face else []

  with MayaProgress("Baking Anny rig", maximum=len(targets) + 4) as prog:
    grp = _Node(mc.group(empty=True, name=name + "_grp"))
    rest = _read(sess, "rig_rest.bin")
    mesh = _Node(_build.build_mesh(topo, rest, name=name))
    mc.parent(mesh.path, grp.path)

    # --- blendShape: one target per ARKit facial action ----------------------
    blend = None
    if targets:
      tmpl = _Node(mc.duplicate(mesh.path, name=name + "_faceTmp")[0])
      mc.parent(tmpl.path, world=True)
      blend = mc.blendShape(mesh.path, name=name + "_faceShapes",
                            frontOfChain=True)[0]
      rest_list = list(rest)
      for i, tgt in enumerate(targets):
        prog.set(i, "Face shape %d/%d: %s" % (i + 1, len(targets), tgt["name"]))
        idx = _read(sess, "rig_face_%02d_idx.bin" % i, "i")
        dlt = _read(sess, "rig_face_%02d.bin" % i)
        pts = list(rest_list)
        for k, v in enumerate(idx):
          pts[3 * v] += dlt[3 * k]
          pts[3 * v + 1] += dlt[3 * k + 1]
          pts[3 * v + 2] += dlt[3 * k + 2]
        tmp = mc.duplicate(tmpl.path, name=tgt["name"])[0]
        _build.set_points(tmp, pts)
        mc.blendShape(blend, edit=True, target=(mesh.path, i, tmp, 1.0))
        # The weight is aliased after the target mesh; fix it when Maya had
        # to uniquify that name (another node already called e.g. jawOpen).
        plug = "%s.w[%d]" % (blend, i)
        if mc.aliasAttr(plug, query=True) != tgt["name"]:
          mc.aliasAttr(tgt["name"], plug)
        mc.delete(tmp)  # deltas are stored on the blendShape node
      mc.delete(tmpl.path)

    # --- joints at the rest (bind) pose ---------------------------------------
    prog.set(len(targets), "Building the 104-bone skeleton…")
    labels = meta["bone_labels"]
    parents = meta["bone_parents"]
    joints = []
    for i, lab in enumerate(labels):
      mc.select(clear=True)
      j = _Node(mc.joint(name=joint_name(lab)))
      mc.parent(j.path, joints[parents[i]].path if parents[i] >= 0
                else grp.path)
      mc.setAttr(j.path + ".radius", 1.5)
      joints.append(j)
    for i, j in enumerate(joints):
      mc.xform(j.path, worldSpace=True, matrix=meta["rest_world"][i])
    # Rest orientation -> jointOrient so the bind pose has zero rotations.
    mc.makeIdentity(joints[0].path, apply=True, rotate=True, translate=False,
                    scale=False)

    # --- skin ------------------------------------------------------------------
    prog.set(len(targets) + 1, "Binding Anny's skin weights…")
    jpaths = [j.path for j in joints]
    skin = mc.skinCluster(jpaths, mesh.path, name=name + "_skinCluster",
                          toSelectedBones=True, bindMethod=0, skinMethod=0,
                          normalizeWeights=1, obeyMaxInfluences=False,
                          maximumInfluences=8)[0]
    _set_skin_weights(skin, mesh.path, jpaths, sess, meta["num_vertices"])

    # --- current pose + expression ----------------------------------------
    prog.set(len(targets) + 2, "Applying pose…")
    if apply_pose:
      set_pose(jpaths, meta["posed_world"])
    if blend:
      for i, tgt in enumerate(targets):
        if tgt["value"]:
          mc.setAttr("%s.w[%d]" % (blend, i), tgt["value"])

    mpath = mesh.path
    mc.addAttr(mpath, longName=RIG_ATTR, dataType="string")
    mc.setAttr(mpath + "." + RIG_ATTR, json.dumps({
        "state": state, "skinCluster": skin, "blendShape": blend,
        "joint_uuids": [j.uuid for j in joints], "bone_labels": labels}),
        type="string")
  mc.select(mpath, replace=True)
  logger.info("Baked Anny rig '%s' (%d joints, %d face shapes)", mpath,
              len(joints), len(targets))
  return mc.ls(mpath)[0]


def rig_info(mesh):
  raw = mc.getAttr(mesh + "." + RIG_ATTR)
  return json.loads(raw) if raw else {}


def rig_joints(mesh):
  """Long names of a baked rig's joints, in Anny bone order."""
  return [mc.ls(u, long=True)[0] for u in rig_info(mesh)["joint_uuids"]]


def set_pose(joints, world_matrices):
  """Pose joints by world matrices (Maya row-vector 4x4, root first)."""
  for j, m in zip(joints, world_matrices):
    mc.xform(j, worldSpace=True, matrix=m)


def _set_skin_weights(skin, mesh, joints, session_dir, num_vertices):
  """Bulk-load Anny's (V, B) weights into the skinCluster via the API."""
  import maya.api.OpenMayaAnim as oma
  nb = len(joints)
  w = _read(session_dir, "rig_weights.bin", "d")
  if len(w) != num_vertices * nb:
    raise ValueError("skin weights size mismatch: %d != %d*%d"
                     % (len(w), num_vertices, nb))
  sel = om2.MSelectionList()
  sel.add(skin)
  sel.add(mesh)
  fn = oma.MFnSkinCluster(sel.getDependNode(0))
  dag = sel.getDagPath(1)
  dag.extendToShape()
  # Influence order in the cluster may differ from our joint order: map it.
  influences = [p.fullPathName() for p in fn.influenceObjects()]
  inf_index = {n: i for i, n in enumerate(influences)}
  order = [inf_index[j] for j in joints]
  if order == list(range(nb)):
    flat = om2.MDoubleArray(w)
  else:
    flat = om2.MDoubleArray(num_vertices * nb, 0.0)
    for v in range(num_vertices):
      base = v * nb
      for b in range(nb):
        flat[base + order[b]] = w[base + b]
  comps = om2.MFnSingleIndexedComponent()
  comp_obj = comps.create(om2.MFn.kMeshVertComponent)
  comps.setCompleteData(num_vertices)
  fn.setWeights(dag, comp_obj, om2.MIntArray(list(range(nb))), flat,
                normalize=False)
