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
  """Constant mesh topology: quads, per-face-vertex UVs, part per face."""

  __slots__ = ("faces", "face_uvs", "face_parts", "meta")

  def __init__(self, faces, face_uvs, face_parts, meta):
    self.faces = faces            # array('i'), flat [a,b,c,d, ...]
    self.face_uvs = face_uvs      # array('f'), flat [u,v, ...] per face-vertex
    self.face_parts = face_parts  # array('i'), part index per face
    self.meta = meta              # topology.json dict

  @property
  def num_faces(self):
    return self.meta["num_faces"]

  @property
  def num_vertices(self):
    return self.meta["num_vertices"]

  @property
  def verts_per_face(self):
    return self.meta["verts_per_face"]


def read_topology(session_dir):
  with open(os.path.join(session_dir, "topology.json")) as f:
    meta = json.load(f)
  faces = read_array(os.path.join(session_dir, "faces.bin"), "i")
  uvs = read_array(os.path.join(session_dir, "face_uvs.bin"), "f")
  parts = read_array(os.path.join(session_dir, "face_parts.bin"), "i")
  n = meta["num_faces"] * meta["verts_per_face"]
  if len(faces) != n or len(uvs) != 2 * n or len(parts) != meta["num_faces"]:
    raise ValueError("topology files truncated in %s" % session_dir)
  return Topology(faces, uvs, parts, meta)


def read_vertices(path):
  """Flat array('f') of [x,y,z, ...]."""
  return read_array(path, "f")
