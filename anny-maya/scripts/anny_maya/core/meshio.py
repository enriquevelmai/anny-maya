"""Pure-Python readers for the worker's binary files (no numpy in mayapy).

Little-endian float32/int32, matching external/server.py's writers.
"""

from __future__ import annotations

import array
import json
import os
import sys


def read_array(path, typecode):
  a = array.array(typecode)
  with open(path, "rb") as f:
    a.frombytes(f.read())
  if sys.byteorder == "big":  # files are little-endian
    a.byteswap()
  return a


class Topology(object):
  """Constant mesh topology: faces (quads, plus the triangles of the GNM
  neck strip), per-face-vertex UVs, part per face."""

  __slots__ = ("faces", "counts", "face_uvs", "face_parts", "part_names",
               "num_vertices", "meta", "kind")

  def __init__(self, faces, counts, face_uvs, face_parts, part_names,
               num_vertices, meta, kind):
    self.faces = faces            # array('i'), flat vertex ids of all faces
    self.counts = counts          # array('i'), vertices per face
    self.face_uvs = face_uvs      # array('f'), flat [u,v, ...] per face-vertex
    self.face_parts = face_parts  # array('i'), part index per face
    self.part_names = part_names  # list of str
    self.num_vertices = num_vertices
    self.meta = meta              # the whole topology.json dict
    self.kind = kind              # "body" or "head" (Anny body + GNM head)

  @property
  def num_faces(self):
    return len(self.counts)


def _read_kind(session_dir, prefix, info, meta, kind):
  faces = read_array(os.path.join(session_dir, prefix + "faces.bin"), "i")
  counts = read_array(os.path.join(session_dir, prefix + "face_counts.bin"),
                      "i")
  uvs = read_array(os.path.join(session_dir, prefix + "face_uvs.bin"), "f")
  parts = read_array(os.path.join(session_dir, prefix + "face_parts.bin"), "i")
  n = sum(counts)
  if (len(counts) != info["num_faces"] or len(faces) != n
      or len(uvs) != 2 * n or len(parts) != len(counts)):
    raise ValueError("topology files truncated in %s" % session_dir)
  return Topology(faces, counts, uvs, parts, list(info["part_names"]),
                  info["num_vertices"], meta, kind)


def read_topology(session_dir):
  """(body topology, combined body+GNM-head topology or None)."""
  with open(os.path.join(session_dir, "topology.json")) as f:
    meta = json.load(f)
  body = _read_kind(session_dir, "", meta, meta, "body")
  head = None
  if meta.get("head"):
    head = _read_kind(session_dir, "head_", meta["head"], meta, "head")
  return body, head


def read_vertices(path):
  """Flat array('f') of [x,y,z, ...]."""
  return read_array(path, "f")
