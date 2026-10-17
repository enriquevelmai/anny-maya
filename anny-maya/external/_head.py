# Copyright 2026 Enrique Velasco Mairal
# The GNM bind-pose evaluation follows gnm/shape/gnm_common.py of Google GNM
# (https://github.com/google/GNM), Copyright 2026 Google LLC, licensed under
# the Apache License, Version 2.0. Modified: re-implemented in numpy and
# extended to attach the head to the NAVER Anny body.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: a Google GNM head on the Anny body (numpy only).

GNM (github.com/google/GNM, Apache-2.0) is a 3D morphable head model: a
template plus linear identity (253) and expression (383) bases over a fixed
17,821-vertex quad mesh (skin, eyes, teeth, tongue). This module:

  * evaluates GNM's bind pose (template + identity + expression deltas) -
    exactly ``gnm.shape.gnm_common.vertex_positions_bind_pose``; GNM's own
    joints/LBS are not needed because the head rides Anny's skeleton;
  * aligns it to the Anny body by a similarity transform (uniform scale from
    the eye distance, eye midpoints coincide; GNM is Y-up/+Z-front, Anny
    Z-up/-Y-front), plus optional user fit offsets;
  * hides Anny's head: Anny faces above one 42-vertex neck edge loop (L_A)
    and Anny's helper eyes/tongue are dropped; GNM faces below one of its
    110-vertex neck rings (L_G) are dropped;
  * conforms GNM's lower neck so L_G lies on Anny's next loop above L_A,
    with a smooth falloff over K rings (per shape, so any body/face works);
  * zips L_A to L_G with a strip of triangles -> one watertight mesh;
  * skins every GNM vertex with weights transferred from Anny's own head and
    neck (eyeballs -> Anny's eye bones), so the whole thing deforms with the
    104-bone ``anny`` rig, poses, and bakes to one skinCluster.

The topology choice (which loops) is made once on the neutral shapes and
stored with the compiled data; per evaluation only positions change.
"""

from __future__ import annotations

import collections
import os

import numpy as np

HEAD_COMPILED_VERSION = 3  # 3: radial surface blend of the neck onto Anny
HEAD_COMPILED_NAME = "head_compiled_v%d.npz" % HEAD_COMPILED_VERSION
GNM_NPZ_REL = os.path.join("gnm", "shape", "data", "versions", "v3_0",
                           "gnm_head.npz")

# GNM mesh component -> material part name used on the Maya side.
GNM_PART = {"skin": "gnm_skin", "left_eye": "gnm_eye_l",
            "right_eye": "gnm_eye_r", "upper_teeth_and_gums": "gnm_teeth",
            "lower_teeth_and_gums": "gnm_teeth", "tongue": "gnm_tongue"}
CONFORM_RINGS = 60       # GNM rings (above the cut) eligible for the neck blend
SURFACE_MATCH_MAX = 0.02  # m, see _surface_correspondence
NAPE_BLEND = 0.08        # metres of neck blend at the back (front: to chin)
UV_TILE_OFFSET = 1.0     # GNM UVs live in the next UDIM tile (u + 1)


def gnm_to_anny(p):
  """GNM frame (m, Y up, +Z front) -> Anny frame (m, Z up, -Y front)."""
  p = np.asarray(p, np.float64)
  return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


# --- GNM model -----------------------------------------------------------------

class GNMModel(object):
  """Minimal numpy GNM: the parts of gnm_head.npz the attachment needs."""

  def __init__(self, npz_path):
    z = np.load(npz_path, allow_pickle=False)
    self.template = z["template_vertex_positions"].astype(np.float64)
    self.joint_template = z["template_joint_positions"].astype(np.float64)
    self.identity_basis = z["vertex_identity_basis"]          # (I,V,3) f32
    self.joint_identity_basis = z["joint_identity_basis"]     # (I,J,3)
    self.expression_basis = z["expression_basis"]             # (E,V,3)
    self.identity_names = [str(x) for x in z["identity_names"]]
    self.expression_names = [str(x) for x in z["expression_names"]]
    self.joint_names = [str(x) for x in z["joint_names"]]
    self.quads = z["quads"].astype(np.int64)
    self.quad_uvs = z["quad_uvs"].astype(np.float64)
    self.component_names = [str(x) for x in z["mesh_component_names"]]
    self.group_names = [str(x) for x in z["vertex_group_names"]]
    self.groups = z["vertex_groups"]
    self.mirror_indices = z["mirror_indices"].astype(np.int64)
    self.num_vertices = len(self.template)
    self.identity_dim = len(self.identity_names)
    self.expression_dim = len(self.expression_names)
    self._id_key = None
    self._id_cache = None

  def group(self, name):
    return self.groups[self.group_names.index(name)]

  def _identity_part(self, identity):
    ident = np.zeros(self.identity_dim)
    if identity is not None and len(identity):
      ident[:len(identity)] = np.asarray(identity, np.float64)
    key = ident.tobytes()
    if key != self._id_key:
      nz = np.nonzero(ident)[0]
      v = self.template.copy()
      j = self.joint_template.copy()
      if len(nz):
        v += np.einsum("i,ivk->vk", ident[nz],
                       self.identity_basis[nz].astype(np.float64))
        j += np.einsum("i,ijk->jk", ident[nz],
                       self.joint_identity_basis[nz].astype(np.float64))
      self._id_key, self._id_cache = key, (v, j)
    return self._id_cache

  def bind(self, identity=None, expression=None):
    """Bind-pose vertices + joints (GNM frame), as gnm_common does."""
    v, j = self._identity_part(identity)
    v = v.copy()
    if expression is not None and len(expression):
      e = np.zeros(self.expression_dim)
      e[:len(expression)] = np.asarray(expression, np.float64)
      nz = np.nonzero(e)[0]
      if len(nz):
        v += np.einsum("i,ivk->vk", e[nz],
                       self.expression_basis[nz].astype(np.float64))
    return v, j


# --- mesh helpers ---------------------------------------------------------------

def _edge_faces(faces):
  ef = collections.defaultdict(list)
  for fi, f in enumerate(faces):
    n = len(f)
    for i in range(n):
      a, b = int(f[i]), int(f[(i + 1) % n])
      ef[(min(a, b), max(a, b))].append(fi)
  return ef


def _neighbors(faces):
  nb = collections.defaultdict(set)
  for f in faces:
    n = len(f)
    for i in range(n):
      a, b = int(f[i]), int(f[(i + 1) % n])
      nb[a].add(b)
      nb[b].add(a)
  return nb


def _walk_loop(a, b, nb, ef, limit=1000):
  """Quad edge loop through edge (a, b): at each valence-4 vertex continue
  along the edge sharing no face with the incoming one."""
  loop = [a, b]
  while True:
    if len(nb[b]) != 4:
      return None
    fs = set(ef[(min(a, b), max(a, b))])
    cand = [n for n in nb[b]
            if n != a and not (set(ef[(min(b, n), max(b, n))]) & fs)]
    if len(cand) != 1:
      return None
    a, b = b, cand[0]
    if b == loop[0]:
      return loop
    loop.append(b)
    if len(loop) > limit:
      return None


def _order_ring(ring, nb):
  """Order a closed ring of vertices (each with 2 ring neighbors)."""
  ring_set = set(ring)
  start = ring[0]
  order = [start]
  prev, cur = None, start
  while True:
    nxt = sorted(n for n in nb[cur] if n in ring_set and n != prev)
    if len(nxt) != (2 if prev is None else 1):
      raise ValueError("ring is not a simple loop")
    prev, cur = cur, nxt[0]
    if cur == start:
      break
    order.append(cur)
    if len(order) > len(ring):
      raise ValueError("ring walk did not close")
  if len(order) != len(ring):
    raise ValueError("ring is not a single loop (%d of %d)"
                     % (len(order), len(ring)))
  return order


class _Frame(object):
  """Cylindrical frame around the neck axis (angle 0 = front)."""

  def __init__(self, base, top):
    self.o = np.asarray(base, np.float64)
    a = np.asarray(top, np.float64) - self.o
    self.axis = a / np.linalg.norm(a)
    front = np.array([0.0, -1.0, 0.0])        # Anny front is -Y
    front -= self.axis * front.dot(self.axis)
    self.u = front / np.linalg.norm(front)
    self.v = np.cross(self.axis, self.u)

  def angle(self, p):
    d = np.asarray(p) - self.o
    return np.arctan2(d @ self.v, d @ self.u)

  def height(self, p):
    return (np.asarray(p) - self.o) @ self.axis


def _interp_periodic(theta_src, values, theta_q):
  """Linear interpolation of ``values`` (n, k) known at angles theta_src
  (any order) at query angles theta_q (periodic in 2*pi)."""
  order = np.argsort(theta_src)
  t = theta_src[order]
  v = values[order]
  t_ext = np.concatenate([t[-1:] - 2 * np.pi, t, t[:1] + 2 * np.pi])
  v_ext = np.concatenate([v[-1:], v, v[:1]])
  q = (theta_q + np.pi) % (2 * np.pi) - np.pi
  out = np.empty((len(q), v.shape[1]))
  for k in range(v.shape[1]):
    out[:, k] = np.interp(q, t_ext, v_ext[:, k])
  return out


# --- compile (topology + weights, once) ----------------------------------------

def compile_attachment(anny, gnm):
  """Choose the cut loops on the neutral shapes, build the combined fixed
  topology, bridge strip and GNM skin weights. ``anny`` is an
  _anny_core.AnnyModel, ``gnm`` a GNMModel. Returns a dict of arrays."""
  c0 = anny.coefficients()
  rv = anny.rest_vertices(c0)
  rp = anny.rest_skeleton(c0)
  lab = {l: i for i, l in enumerate(anny.bone_labels)}
  pos = lambda b: rp[lab[b], :3, 3]

  # ---- Anny cut loop L_A: the 42-vertex neck loop closest to mid-neck.
  body_faces = anny.faces[anny.face_part == 0]
  nb = _neighbors(body_faces)
  ef = _edge_faces(body_faces)
  zmid = 0.5 * (pos("neck01")[2] + pos("neck02")[2])
  best = None
  for v in np.nonzero((np.abs(rv[:, 0]) < 0.006)
                      & (np.abs(rv[:, 2] - zmid) < 0.03)
                      & (rv[:, 1] < pos("neck02")[1]))[0]:
    for w in nb[int(v)]:
      if abs(rv[w, 2] - rv[v, 2]) > 0.5 * np.linalg.norm(rv[w] - rv[v]):
        continue
      lp = _walk_loop(int(v), int(w), nb, ef)
      if lp is None or not (30 <= len(lp) <= 60):
        continue
      score = abs(rv[lp, 2].mean() - zmid) + np.ptp(rv[lp, 2]) * 0.1
      if best is None or score < best[0]:
        best = (score, lp)
  if best is None:
    raise RuntimeError("could not find Anny's neck edge loop")
  loop_a = best[1]

  # Head side: flood fill body faces from the crown, never crossing L_A.
  la_set = set(loop_a)
  la_edges = {(min(a, b), max(a, b)) for a, b in
              zip(loop_a, loop_a[1:] + loop_a[:1])}
  crown = int(np.argmax(np.where(np.isin(np.arange(len(rv)),
                                         np.unique(body_faces)), rv[:, 2],
                                 -1e9)))
  face_of_vertex = collections.defaultdict(list)
  for fi, f in enumerate(body_faces):
    for x in f:
      face_of_vertex[int(x)].append(fi)
  head_faces = set()
  stack = [face_of_vertex[crown][0]]
  while stack:
    fi = stack.pop()
    if fi in head_faces:
      continue
    head_faces.add(fi)
    f = body_faces[fi]
    for i in range(4):
      a, b = int(f[i]), int(f[(i + 1) % 4])
      e = (min(a, b), max(a, b))
      if e in la_edges:
        continue
      for g in ef[e]:
        if g not in head_faces:
          stack.append(g)
  keep_body = np.array([i not in head_faces for i in range(len(body_faces))])
  hf = body_faces[np.array(sorted(head_faces))]
  if rv[hf, 2].min() < rv[loop_a, 2].min() - 0.03:
    raise RuntimeError("neck cut leaked into the body")

  # Next two loops above L_A (1:1 over the quad band) for the conform target.
  def loop_above(loop, below):
    out = []
    for x in loop:
      cand = [n for n in nb[x] if n not in set(loop) and n not in below]
      cand = [n for n in cand if any(fi in head_faces
                                     for fi in face_of_vertex[n])]
      if len(cand) != 1:
        raise RuntimeError("irregular band above the neck loop")
      out.append(cand[0])
    return out
  loop_a1 = loop_above(loop_a, set())
  loop_a2 = loop_above(loop_a1, la_set)

  # ---- GNM: align template, rings from its neck boundary.
  gv, gj = gnm.bind()
  align = similarity(anny, rv, rp, gv, gj)
  G = apply_similarity(align, gv)
  skin = gnm.group("skin") > 0.5
  quads = gnm.quads
  skin_q = skin[quads].all(axis=1)
  gnb = _neighbors(quads[skin_q])
  gef = _edge_faces(quads[skin_q])
  boundary = [e for e, fs in gef.items() if len(fs) == 1]
  bverts = {v for e in boundary for v in e}
  dist = {v: 0 for v in bverts}
  front = list(bverts)
  while front:
    nxt = []
    for v in front:
      for n in gnb[v]:
        if n not in dist:
          dist[n] = dist[v] + 1
          nxt.append(n)
    front = nxt
  rings = collections.defaultdict(list)
  for v, d in dist.items():
    rings[d].append(v)
  frame = _Frame(pos("neck01"), pos("head"))
  a1_h = frame.height(rv[loop_a1]).mean()
  ring_k = None
  for k in sorted(rings):
    r = rings[k]
    if len(r) != len(rings[0]):
      break
    if frame.height(G[r]).mean() >= a1_h:
      ring_k = k
      break
  if ring_k is None:
    raise RuntimeError("no GNM neck ring above Anny's cut")
  loop_g = _order_ring(rings[ring_k], gnb)

  # ---- keep sets
  gnm_drop_v = np.zeros(gnm.num_vertices, bool)
  for v, d in dist.items():
    if d < ring_k:
      gnm_drop_v[v] = True
  keep_gq = ~gnm_drop_v[quads].any(axis=1)
  # Non-skin quads (eyes/teeth/tongue) are never below the neck ring.
  comp_of_q = np.full(len(quads), -1)
  for ci, name in enumerate(gnm.component_names):
    mask = gnm.group(name) > 0.5
    comp_of_q[mask[quads].all(axis=1) & (comp_of_q < 0)] = ci

  body_keep_faces = body_faces[keep_body]
  anny_used = np.unique(body_keep_faces)
  gnm_used = np.unique(quads[keep_gq])
  na = len(anny_used)
  a_map = np.full(len(rv), -1, np.int64)
  a_map[anny_used] = np.arange(na)
  g_map = np.full(gnm.num_vertices, -1, np.int64)
  g_map[gnm_used] = na + np.arange(len(gnm_used))

  # UVs (per face-vertex)
  anny_body_uv = anny.face_uv[anny.face_part == 0][keep_body]
  uv_a = anny.uvs[anny_body_uv]                       # (Fa, 4, 2)
  uv_g = gnm.quad_uvs[keep_gq].copy()
  uv_g[..., 0] += UV_TILE_OFFSET

  # ---- bridge strip L_A -> L_G (triangles), consistent winding.
  th_a = frame.angle(rv[loop_a])
  th_g = frame.angle(G[loop_g])
  la = _ccw(loop_a, th_a)
  lg = _ccw(loop_g, th_g)
  th_a = frame.angle(rv[la])
  th_g = frame.angle(G[lg])
  tris = _zip(la, th_a, lg, th_g)
  # Orientation: the strip must traverse the L_A edges opposite to the kept
  # Anny faces that own them.
  owner = None
  for f in body_keep_faces:
    fl = [int(x) for x in f]
    for i in range(4):
      if fl[i] == la[0] and fl[(i + 1) % 4] == la[1]:
        owner = True
      if fl[i] == la[1] and fl[(i + 1) % 4] == la[0]:
        owner = False
  flip = owner is True
  # uv for strip corners: Anny's uv of each L_A vertex; GNM's for L_G.
  vuv_a = {}
  for f, fuv in zip(body_keep_faces, uv_a):
    for x, u in zip(f, fuv):
      vuv_a.setdefault(int(x), u)
  vuv_g = {}
  for f, fuv in zip(quads[keep_gq], uv_g):
    for x, u in zip(f, fuv):
      vuv_g.setdefault(int(x), u)
  strip, strip_uv = [], []
  for t in tris:
    (s0, i0), (s1, i1), (s2, i2) = t
    tri = [(s0, i0), (s1, i1), (s2, i2)]
    if flip:
      tri = tri[::-1]
    strip.append([a_map[i] if s == "a" else g_map[i] for s, i in tri])
    strip_uv.append([vuv_a[i] if s == "a" else vuv_g[i] for s, i in tri])

  # ---- weights for GNM vertices: IDW from Anny head/neck verts (neutral).
  W = anny.weights
  near = np.unique(body_faces[np.array(sorted(head_faces))])
  near = np.unique(np.concatenate([near, anny_used[rv[anny_used, 2] >
                                                   pos("neck01")[2] - 0.05]]))
  gw = np.zeros((len(gnm_used), W.shape[1]))
  P = rv[near]
  Gu = G[gnm_used]
  for s in range(0, len(Gu), 2048):
    d2 = ((Gu[s:s + 2048, None, :] - P[None]) ** 2).sum(-1)
    k = np.argpartition(d2, 6, axis=1)[:, :6]
    dk = np.take_along_axis(d2, k, axis=1)
    w = 1.0 / (np.sqrt(dk) + 1e-4) ** 2
    w /= w.sum(1, keepdims=True)
    gw[s:s + 2048] = np.einsum("nk,nkb->nb", w, W[near[k]])
  eye_l = gnm.group("left_eye")[gnm_used] > 0.5
  eye_r = gnm.group("right_eye")[gnm_used] > 0.5
  gw[eye_l] = 0.0
  gw[eye_l, lab["eye.L"]] = 1.0
  gw[eye_r] = 0.0
  gw[eye_r, lab["eye.R"]] = 1.0
  gw /= gw.sum(1, keepdims=True)

  # ---- conform data: kept skin verts on the neck (between the cut ring and
  # the chin); weighted per evaluation by height.
  dist_g = np.full(gnm.num_vertices, 10 ** 6)
  for v, d in dist.items():
    dist_g[v] = d - ring_k
  face_mask = gnm.group("hockey_mask") > 0.5
  conform_v = np.nonzero((dist_g >= 0) & (dist_g <= CONFORM_RINGS)
                         & ~face_mask & ~(gnm.group("ears") > 0.5))[0]
  conform_w = np.zeros(len(conform_v))  # unused (height-based at runtime)
  chin_v = np.nonzero(gnm.group("chin_region") > 0.5)[0]

  # ---- faces, parts, uvs of the combined mesh
  faces = ([[int(a_map[x]) for x in f] for f in body_keep_faces]
           + [[int(g_map[x]) for x in f] for f in quads[keep_gq]]
           + strip)
  fuv = ([list(map(tuple, u)) for u in uv_a] + [list(map(tuple, u))
                                                 for u in uv_g]
         + [list(map(tuple, u)) for u in strip_uv])
  parts = ["body"] + sorted(set(GNM_PART.values()))
  part_idx = {p: i for i, p in enumerate(parts)}
  fpart = ([0] * len(body_keep_faces)
           + [part_idx[GNM_PART.get(gnm.component_names[c], "gnm_skin")]
              if c >= 0 else part_idx["gnm_skin"] for c in comp_of_q[keep_gq]]
           + [0] * len(strip))
  counts = np.array([len(f) for f in faces], np.int32)
  return dict(
      version=HEAD_COMPILED_VERSION,
      anny_used=anny_used.astype(np.int64),
      gnm_used=gnm_used.astype(np.int64),
      face_counts=counts,
      faces_flat=np.concatenate([np.asarray(f, np.int64) for f in faces]),
      face_uvs_flat=np.concatenate([np.asarray(u, np.float64).reshape(-1, 2)
                                    for u in fuv]),
      face_part=np.asarray(fpart, np.int32),
      part_names=np.asarray(parts),
      loop_a=np.asarray(la, np.int64), loop_a1=np.asarray(
          [loop_a1[loop_a.index(x)] for x in la], np.int64),
      loop_g=np.asarray(lg, np.int64),
      conform_v=conform_v.astype(np.int64),
      conform_w=conform_w,
      gnm_weights=gw,
      ring_k=np.asarray(ring_k),
      chin_v=chin_v.astype(np.int64),
      **_surface_correspondence(rv, body_faces, frame, G, conform_v),
  )


def _surface_correspondence(rv, body_faces, frame, G, verts):
  """For each GNM neck vertex, the point of Anny's ORIGINAL neck/nape
  surface (hidden head faces included) hit by a ray from the neck axis
  through that vertex, as (triangle, barycentrics) on Anny's mesh, so it
  follows any body shape at runtime. Neutral shapes; computed once."""
  tris = np.concatenate([body_faces[:, [0, 1, 2]], body_faces[:, [0, 2, 3]]])
  P = G[verts]
  h = frame.height(P)
  th = frame.height(rv[tris])
  near = (th.max(axis=1) > h.min() - 0.03) & (th.min(axis=1) < h.max() + 0.03)
  tris = tris[near]
  v0, v1, v2 = rv[tris[:, 0]], rv[tris[:, 1]], rv[tris[:, 2]]
  e1, e2 = v1 - v0, v2 - v0
  origin = frame.o + h[:, None] * frame.axis
  d = P - origin
  d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
  tri_out = np.zeros((len(P), 3), np.int64)
  bary_out = np.zeros((len(P), 3))
  ok = np.zeros(len(P), bool)
  for s in range(0, len(P), 256):
    o, dd = origin[s:s + 256], d[s:s + 256]
    pv = np.cross(dd[:, None, :], e2[None])                 # (r, t, 3)
    det = np.einsum("tk,rtk->rt", e1, pv)
    inv = 1.0 / np.where(np.abs(det) < 1e-12, np.nan, det)
    tv = o[:, None, :] - v0[None]
    u = np.einsum("rtk,rtk->rt", tv, pv) * inv
    qv = np.cross(tv, e1[None])
    w = np.einsum("rk,rtk->rt", dd, qv) * inv
    t = np.einsum("tk,rtk->rt", e2, qv) * inv
    hit = (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-6)
    t = np.where(hit, t, np.inf)
    j = np.argmin(t, axis=1)                                # first hit
    good = np.isfinite(t[np.arange(len(j)), j])
    rows = np.arange(len(j))
    tri_out[s:s + 256] = tris[j]
    bu, bw = u[rows, j], w[rows, j]
    bary_out[s:s + 256] = np.stack([1 - bu - bw, bu, bw], axis=1)
    ok[s:s + 256] = good
  # Trust only correspondences where the two necks nearly coincide on the
  # neutral shapes (tube-like neck): under the chin the ray can hit an
  # unrelated part of Anny's jaw.
  hit_p = np.einsum("nk,nkc->nc", bary_out, rv[tri_out])
  ok &= np.linalg.norm(hit_p - P, axis=1) < SURFACE_MATCH_MAX
  bary_out[~ok] = 0.0
  return {"surf_tri": tri_out, "surf_bary": bary_out, "surf_ok": ok}


def _ccw(loop, theta):
  """Rotate/reverse a closed loop so its angles increase."""
  n = len(loop)
  inc = sum(((theta[(i + 1) % n] - theta[i] + np.pi) % (2 * np.pi) - np.pi)
            for i in range(n))
  loop = list(loop) if inc > 0 else list(loop)[::-1]
  return loop


def _zip(la, ta, lg, tg):
  """Triangle strip between two closed loops ordered by increasing angle
  (rotationally merged by angle). Returns [((side, idx) x3), ...]."""
  def unwrap(t):
    t = np.array(t, np.float64)
    out = [t[0]]
    for x in t[1:]:
      d = (x - out[-1] + np.pi) % (2 * np.pi) - np.pi
      out.append(out[-1] + d)
    return np.array(out)
  ua = unwrap(ta)
  ug = unwrap(tg)
  # start both at the same angle: rotate the GNM loop to its closest point
  j0 = int(np.argmin(np.abs((ug - ua[0] + np.pi) % (2 * np.pi) - np.pi)))
  lg = list(lg[j0:]) + list(lg[:j0])
  ug = unwrap(list(np.array(tg)[j0:]) + list(np.array(tg)[:j0]))
  ug = ug - 2 * np.pi * np.round((ug[0] - ua[0]) / (2 * np.pi))
  na, ng = len(la), len(lg)
  i = j = 0
  tris = []
  while i < na or j < ng:
    a0, a1 = la[i % na], la[(i + 1) % na]
    g0, g1 = lg[j % ng], lg[(j + 1) % ng]
    ta_next = ua[0] + 2 * np.pi if i + 1 >= na else ua[i + 1]
    tg_next = ug[0] + 2 * np.pi if j + 1 >= ng else ug[j + 1]
    if j >= ng or (i < na and ta_next <= tg_next):
      tris.append((("a", a0), ("a", a1), ("g", g0)))
      i += 1
    else:
      tris.append((("a", a0), ("g", g1), ("g", g0)))
      j += 1
  return tris


# --- per evaluation -------------------------------------------------------------

def similarity(anny, rest_v, rest_p, gnm_v, gnm_j, fit=None):
  """GNM -> Anny-frame similarity (scale, rotation, translation) for this
  body + face: eye midpoints coincide, scale = eye-distance ratio. ``fit``
  (user) = {"scale": mult, "offset": [x,y,z] metres (Anny frame),
  "pitch": degrees about the eye axis}."""
  lab = {l: i for i, l in enumerate(anny.bone_labels)}
  el, er = rest_p[lab["eye.L"], :3, 3], rest_p[lab["eye.R"], :3, 3]
  gj = gnm_to_anny(gnm_j)
  gl, gr = gj[2], gj[3]                         # left_eye, right_eye
  s = np.linalg.norm(el - er) / np.linalg.norm(gl - gr)
  fit = fit or {}
  s *= float(fit.get("scale", 1.0))
  pitch = np.deg2rad(float(fit.get("pitch", 0.0)))
  c, sn = np.cos(pitch), np.sin(pitch)
  rot = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
  t = 0.5 * (el + er) + np.asarray(fit.get("offset", [0, 0, 0]), np.float64)
  return {"s": s, "R": rot, "src": 0.5 * (gl + gr), "dst": t}


def apply_similarity(sim, gnm_points):
  p = gnm_to_anny(gnm_points) - sim["src"]
  return (p @ sim["R"].T) * sim["s"] + sim["dst"]


class HeadAttachment(object):

  def __init__(self, anny, gnm, compiled, arkit=None):
    """``gnm``: a GNMModel, or a zero-arg callable returning one (loaded on
    first use - the bases are ~140 MB, only needed once a head is shown).
    ``arkit``: an _arkit.ArkitDeltas or a zero-arg callable returning one
    (the 52 ARKit facial actions on the GNM head), or None."""
    self.anny = anny
    self._gnm = gnm
    self._arkit = arkit
    c = compiled
    self.anny_used = c["anny_used"]
    self.gnm_used = c["gnm_used"]
    self.face_counts = c["face_counts"]
    self.faces_flat = c["faces_flat"]
    self.face_uvs_flat = c["face_uvs_flat"]
    self.face_part = c["face_part"]
    self.part_names = [str(x) for x in c["part_names"]]
    self.loop_a1 = c["loop_a1"]
    self.loop_g = c["loop_g"]
    self.conform_v = c["conform_v"]
    self.chin_v = c["chin_v"]
    self.surf_tri = c["surf_tri"]
    self.surf_bary = c["surf_bary"]
    self.surf_ok = c["surf_ok"]
    self._ring = np.isin(self.conform_v, self.loop_g)
    self.weights = np.concatenate([anny.weights[self.anny_used],
                                   c["gnm_weights"]])
    self.num_vertices = len(self.anny_used) + len(self.gnm_used)
    lab = {l: i for i, l in enumerate(anny.bone_labels)}
    self._neck = (lab["neck01"], lab["head"])

  @property
  def gnm(self):
    if callable(self._gnm):
      self._gnm = self._gnm()
    return self._gnm

  @property
  def arkit(self):
    if callable(self._arkit):
      self._arkit = self._arkit()
    return self._arkit

  def head_vertices(self, rest_v, rest_p, identity=None, expression=None,
                    fit=None, face=None):
    """Aligned + neck-conformed GNM vertices (Anny rest frame, all V_gnm).
    ``face``: {ARKit name: weight}, added to GNM's bind pose.

    The neck conform is computed on the neutral face (identity only) and
    added as is, so expressions and ARKit actions stay exactly linear: a
    baked blendShape target reproduces the slider (jawOpen moves the chin,
    which the conform's blend span would otherwise follow)."""
    gv0, gj = self.gnm.bind(identity, None)
    gv = gv0
    if expression is not None and any(expression):
      gv, _gj = self.gnm.bind(identity, expression)
    if face and any(face.values()):
      if self.arkit is None:
        raise RuntimeError("ARKit shapes for the GNM head are unavailable "
                           "(Head tab > Use GNM head reinstalls them).")
      d = self.arkit.delta(face)
      if d is not None:
        gv = gv + d
    sim = similarity(self.anny, rest_v, rest_p, gv, gj, fit)
    G = apply_similarity(sim, gv)
    G0 = G if gv is gv0 else apply_similarity(sim, gv0)
    G[self.conform_v] += self._conform(rest_v, rest_p, G0)
    return G

  def _conform(self, rest_v, rest_p, G):
    """Displacement of the conform vertices that lands GNM's cut ring on
    Anny's loop above the cut, fading up the neck."""
    frame = _Frame(rest_p[self._neck[0], :3, 3], rest_p[self._neck[1], :3, 3])
    # target: Anny's loop just above the cut, at each L_G vertex's angle
    target_pts = rest_v[self.loop_a1]
    ta = frame.angle(target_pts)
    lgp = G[self.loop_g]
    tg = frame.angle(lgp)
    tgt = _interp_periodic(ta, target_pts, tg)
    delta = tgt - lgp
    cv = self.conform_v
    dv = _interp_periodic(tg, delta, frame.angle(G[cv]))
    # Blend from full (at the cut ring) to zero just under the chin, by
    # height along the neck axis: smooth front and back alike.
    h = frame.height(G[cv])
    h0 = _interp_periodic(tg, frame.height(lgp)[:, None],
                          frame.angle(G[cv]))[:, 0]   # cut height per angle
    # Same blend span at every angle: cut -> just under the chin at the
    # front, and as far up the nape at the back (the cut is slanted).
    front = np.abs(tg) < 0.5
    span = (frame.height(G[self.chin_v]).min() - 0.005
            - frame.height(lgp[front]).mean())
    span_f = max(span, 0.03)
    span_b = max(span_f, NAPE_BLEND)
    ang = frame.angle(G[cv])
    span_v = span_f + (span_b - span_f) * 0.5 * (1.0 - np.cos(ang))
    t = np.clip((h0 + span_v - h) / span_v, 0.0, 1.0)
    w = t * t * (3 - 2 * t)
    d1 = w[:, None] * dv
    # Surface blend: near the cut the GNM neck takes Anny's own neck/nape
    # surface (shape AND slope), fading to GNM's shape up the blend span -
    # a translation alone kept GNM's slope and left a ledge on wide necks.
    # The cut ring itself stays exactly on Anny's loop (d1).
    A = np.einsum("nk,nkc->nc", self.surf_bary, rest_v[self.surf_tri])
    ws = w * self.surf_ok * ~self._ring
    G1 = G[cv] + d1
    # Radial only (same angle and height about the neck axis): the radius
    # goes to Anny's, the vertex never slides along the surface.
    def radial(p):
      d = p - frame.o
      d = d - np.outer(d @ frame.axis, frame.axis)
      r = np.linalg.norm(d, axis=1)
      return d / np.maximum(r, 1e-9)[:, None], r
    u, r1 = radial(G1)
    _ua, ra = radial(A)
    return d1 + (ws * (ra - r1))[:, None] * u

  def rest_vertices(self, rest_v, rest_p, identity=None, expression=None,
                    fit=None, face=None):
    G = self.head_vertices(rest_v, rest_p, identity, expression, fit, face)
    return np.concatenate([rest_v[self.anny_used], G[self.gnm_used]])

  def skin(self, rest_combined, bone_transforms):
    nb = bone_transforms.shape[0]
    blended = self.weights @ bone_transforms[:, :3, :].reshape(nb, 12)
    blended = blended.reshape(-1, 3, 4)
    return (np.einsum("vij,vj->vi", blended[:, :, :3], rest_combined)
            + blended[:, :, 3])

  def gnm_slice(self):
    """Index range of the GNM vertices inside the combined array."""
    return len(self.anny_used), self.num_vertices


def save_head_compiled(d, path):
  tmp = path + ".tmp.npz"
  np.savez_compressed(tmp, **d)
  os.replace(tmp, path)


def load_head_compiled(path):
  with np.load(path, allow_pickle=False) as z:
    d = {k: z[k] for k in z.files}
  if int(d["version"]) != HEAD_COMPILED_VERSION:
    raise ValueError("stale compiled head")
  return d


def ensure_head(anny, gnm_npz, cache_dir, log=None, arkit=None):
  """HeadAttachment for this install (topology compiled once and cached;
  the GNM bases themselves load lazily). ``arkit(gnm_loader)`` returns the
  ARKit deltas (called lazily, on the first facial action)."""
  path = os.path.join(cache_dir, HEAD_COMPILED_NAME)
  holder = {}

  def gnm_loader():
    if "gnm" not in holder:
      holder["gnm"] = GNMModel(gnm_npz)
    return holder["gnm"]

  def arkit_loader():
    return arkit(lambda: head.gnm) if arkit else None

  compiled = None
  if os.path.isfile(path):
    try:
      compiled = load_head_compiled(path)
    except Exception:
      compiled = None
  if compiled is None:
    if log:
      log("Compiling the GNM head attachment (one time)...")
    save_head_compiled(compile_attachment(anny, gnm_loader()), path)
    compiled = load_head_compiled(path)
  head = HeadAttachment(anny, gnm_loader, compiled, arkit_loader)
  return head


def gnm_names(gnm_npz):
  """Identity/expression mode names without loading the bases."""
  with np.load(gnm_npz, allow_pickle=False) as z:
    return ([str(x) for x in z["identity_names"]],
            [str(x) for x in z["expression_names"]])


def group_ranges(names):
  """Contiguous [label, start, end] per name prefix (e.g. 'head_017')."""
  out = []
  for i, n in enumerate(names):
    prefix = n.rsplit("_", 1)[0]
    if out and out[-1][0] == prefix:
      out[-1][2] = i
    else:
      out.append([prefix, i, i])
  return out
