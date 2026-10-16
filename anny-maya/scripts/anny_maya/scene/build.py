"""Build the quad Maya mesh from Anny's topology, with UVs and materials.

Fixed topology means we build once, then animate via :func:`set_points`.
"""

from __future__ import annotations

import maya.api.OpenMaya as om2
from maya import cmds as mc

from anny_maya.scene import material


def _points(verts_flat):
  # Building the MPointArray from a list of (x,y,z) tuples in one call is much
  # faster than appending MPoint objects in a Python loop (matters per drag).
  xs = verts_flat[0::3]
  ys = verts_flat[1::3]
  zs = verts_flat[2::3]
  return om2.MPointArray(list(zip(xs, ys, zs)))


def build_mesh(topology, verts_flat, name="anny_body", materials=True):
  """Create the quad mesh, assign UVs + per-part materials, soft-shade.

  Returns the transform name.
  """
  nf = topology.num_faces
  k = topology.verts_per_face
  points = _points(verts_flat)
  poly_counts = om2.MIntArray(nf, k)
  poly_connects = om2.MIntArray(list(topology.faces))

  fn = om2.MFnMesh()
  fn.create(points, poly_counts, poly_connects)

  # UVs: one UV per face-vertex (handles seams correctly).
  uv = topology.face_uvs
  us = om2.MFloatArray(list(uv[0::2]))
  vs = om2.MFloatArray(list(uv[1::2]))
  fn.setUVs(us, vs)
  fn.assignUVs(poly_counts, om2.MIntArray(list(range(nf * k))))

  shape_path = fn.getPath()
  shape_path.pop()
  transform = om2.MFnDagNode(shape_path.node()).name()
  transform = mc.rename(transform, name)
  shape = (mc.listRelatives(transform, shapes=True) or [None])[0]
  if shape:
    mc.rename(shape, transform.split("|")[-1] + "Shape")

  if materials:
    assign_materials(transform, topology)
  mc.polySoftEdge(transform, angle=180, constructionHistory=False)
  mc.select(transform, replace=True)
  return transform


def assign_materials(transform, topology):
  parts = topology.meta["part_names"]
  by_part = {}
  for i, p in enumerate(topology.face_parts):
    by_part.setdefault(parts[p], []).append(i)
  for part, idx in by_part.items():
    sg = material.get_or_create(material.MATERIAL_OF_PART.get(part, "skin"))
    mc.sets(_face_ranges(transform, idx), edit=True, forceElement=sg)


def _face_ranges(transform, indices):
  """Compact 'f[a:b]' component strings for a sorted index list."""
  out = []
  start = prev = None
  for i in sorted(indices):
    if start is None:
      start = prev = i
    elif i == prev + 1:
      prev = i
    else:
      out.append("%s.f[%d:%d]" % (transform, start, prev))
      start = prev = i
  if start is not None:
    out.append("%s.f[%d:%d]" % (transform, start, prev))
  return out


def set_points(transform, verts_flat):
  """Push new vertex positions into an existing Anny mesh."""
  sel = om2.MSelectionList()
  sel.add(transform)
  fn = om2.MFnMesh(sel.getDagPath(0))
  n = len(verts_flat) // 3
  if fn.numVertices != n:
    raise ValueError("vertex count mismatch: %d vs %d" % (fn.numVertices, n))
  fn.setPoints(_points(verts_flat), om2.MSpace.kObject)


def get_points(transform, world=False):
  """Flat list of the mesh's vertex positions (object or world space)."""
  sel = om2.MSelectionList()
  sel.add(transform)
  fn = om2.MFnMesh(sel.getDagPath(0))
  pts = fn.getPoints(om2.MSpace.kWorld if world else om2.MSpace.kObject)
  out = []
  for p in pts:
    out.extend((p.x, p.y, p.z))
  return out
