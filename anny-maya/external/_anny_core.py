"""Numpy port of NAVER's Anny body model (rig ``anny``, topology ``anny-quads``).

Reproduces ``anny.Anny(rig="anny", topology="anny-quads",
local_changes="default", facial_actions="all")`` from the official package
(pinned release, see ANNY_DATA.json) without PyTorch:

  build_model(data_dir)  parse the CC0 MakeHuman/MPFB2 assets (base.obj,
                         ``*.target.gz`` macro/local targets, the 52 Face
                         Units ``.target`` files, rig + skin-weight JSON) and
                         the precomputed ``cached/anny.pth`` rig orientation
                         cache, exactly as ``anny.models.full_model`` does.
  AnnyModel.coefficients phenotype/local-change/facial-action -> blend shape
                         weights (MakeHuman macro interpolation).
  AnnyModel.evaluate     blend shapes -> rest mesh + rest skeleton (cached
                         Procrustes orientations) -> forward kinematics
                         (``local-ref`` pose parameterization) -> LBS.

Coordinates follow Anny: metres, Z up (MakeHuman decimetres Y-up rotated
+90 deg about X). ``to_maya`` converts to Maya's Y-up centimetres.

Validated vertex-for-vertex against the official torch package by
``tests/validate_vs_official.py`` (dev-only venv, never shipped).
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np

from _pth_reader import load_pth

# --- constants mirrored from anny.models.model_data -------------------------

PHENOTYPE_VARIATIONS = dict(
    race=["african", "asian", "caucasian"],
    gender=["male", "female"],
    age=["newborn", "baby", "child", "young", "old"],
    muscle=["minmuscle", "averagemuscle", "maxmuscle"],
    weight=["minweight", "averageweight", "maxweight"],
    height=["minheight", "maxheight"],
    proportions=["idealproportions", "uncommonproportions"],
    cupsize=["mincup", "averagecup", "maxcup"],
    firmness=["minfirmness", "averagefirmness", "maxfirmness"],
)
# Phenotype scalars exposed to users ("all" set; "default" hides the last 5).
PHENOTYPE_LABELS = ["gender", "age", "muscle", "weight", "height",
                    "proportions", "cupsize", "firmness",
                    "african", "asian", "caucasian"]
DEFAULT_PHENOTYPE_LABELS = PHENOTYPE_LABELS[:6]

FACIAL_ACTION_LABELS = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]

_TONGUE_BONES = {"tongue00", "tongue01", "tongue02", "tongue03", "tongue04",
                 "tongue05.L", "tongue05.R", "tongue06.L", "tongue06.R",
                 "tongue07.L", "tongue07.R"}
_BREAST_BONES = {"breast.L", "breast.R"}
_FACIAL_EXPRESSION_BONES = {
    "jaw", "special04", "oris02", "oris01", "oris06.L", "oris07.L",
    "oris06.R", "oris07.R", "levator02.L", "levator03.L", "levator04.L",
    "levator05.L", "levator02.R", "levator03.R", "levator04.R", "levator05.R",
    "special01", "oris04.L", "oris03.L", "oris04.R", "oris03.R", "oris06",
    "oris05", "special03", "levator06.L", "levator06.R", "special06.L",
    "special05.L", "orbicularis03.L", "orbicularis04.L", "special06.R",
    "special05.R", "orbicularis03.R", "orbicularis04.R", "temporalis01.L",
    "oculi02.L", "oculi01.L", "temporalis01.R", "oculi02.R", "oculi01.R",
    "temporalis02.L", "risorius02.L", "risorius03.L", "temporalis02.R",
    "risorius02.R", "risorius03.R"}
_ZERO_WEIGHT_BONES = {
    "oris02", "oris06.L", "oris06.R", "levator02.L", "levator03.L",
    "levator04.L", "levator02.R", "levator03.R", "levator04.R", "special01",
    "oris04.L", "oris04.R", "oris06", "special03", "special06.L",
    "special06.R", "temporalis01.L", "oculi02.L", "temporalis01.R",
    "oculi02.R", "temporalis02.L", "risorius02.L", "temporalis02.R",
    "risorius02.R"}
# rig "anny" = MakeHuman default rig minus these (notongue, nobreasts,
# nofacialexpression, pruned).
ANNY_BONES_TO_REMOVE = (_TONGUE_BONES | _BREAST_BONES
                        | _FACIAL_EXPRESSION_BONES | _ZERO_WEIGHT_BONES)

# Anny's nudity edit (anny.models.full_model.get_edited_mesh_faces).
_DISCARD_VERTS = np.concatenate([np.arange(1778, 1794), np.arange(8450, 8466)])
_CAP_L = [[8437, 8438, 8439, 8440], [8436, 8437, 8440, 8441],
          [8435, 8436, 8441, 8442], [8434, 8435, 8442, 8443],
          [8449, 8434, 8443, 8444], [8448, 8449, 8444, 8445],
          [8447, 8448, 8445, 8446]]
_CAP_R = [[1762, 1771, 1770, 1763], [1763, 1770, 1769, 1764],
          [1764, 1769, 1768, 1765], [1765, 1768, 1767, 1766],
          [1762, 1777, 1772, 1771], [1777, 1776, 1773, 1772],
          [1776, 1775, 1774, 1773]]

COMPILED_VERSION = 1
COMPILED_NAME = "anny_compiled_v%d.npz" % COMPILED_VERSION

# Anny frame (m, Z up) -> Maya frame (cm, Y up):  (x, y, z) -> 100*(x, z, -y)
MAYA_BASIS = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0]])
MAYA_SCALE = 100.0


# --- source parsing ---------------------------------------------------------

def _world_matrix():
  """0.1 * Rx(90 deg): MakeHuman decimetres Y-up -> metres Z-up (as roma)."""
  a = np.deg2rad(90.0)
  c, s = np.cos(a), np.sin(a)
  return 0.1 * np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _apply(m, x):
  return x @ m.T


def load_obj(path):
  """anny.utils.obj_utils.load_obj_file (same group/face semantics)."""
  vertices, uvs = [], []
  groups = {}
  name = "noname"
  fvi, fti = [], []
  with open(path, "r") as f:
    for line in f.readlines():
      line = line.strip()
      if len(line) > 0 and not line.startswith("#") or line.startswith("mtllib"):
        split = line.split(" ")
        tag = split[0]
        if tag == "o":
          if vertices:
            break
        elif tag == "v":
          vertices.append([float(x) for x in split[1:]])
        elif tag == "vt":
          uvs.append([float(x) for x in split[1:]])
        elif tag == "g":
          if fvi:
            groups[name] = (fvi, fti)
          name = split[1]
          if name in groups:
            fvi, fti = list(groups[name][0]), list(groups[name][1])
          else:
            fvi, fti = [], []
        elif tag == "f":
          vids, vtids = [], []
          for x in split[1:]:
            d = [int(y) - 1 for y in x.split("/")]
            vids.append(d[0])
            if len(d) == 2:
              vtids.append(d[1])
          fvi.append(vids)
          fti.append(vtids)
  if fvi:
    groups[name] = (fvi, fti)
  out = {}
  for k, (a, b) in groups.items():
    out[k] = (np.asarray(a, np.int64), np.asarray(b, np.int64))
  return np.asarray(vertices, np.float64), np.asarray(uvs, np.float64), out


def _parse_target_text(text, nverts):
  """Rows 'idx dx dy dz' -> (idx int64[n], delta float64[n,3]); later rows
  overwrite earlier ones (as the official dense assignment does)."""
  rows = [ln.split() for ln in text.splitlines()]
  rows = [r for r in rows if r and not r[0].startswith("#")]
  if not rows:
    return np.zeros(0, np.int64), np.zeros((0, 3))
  for r in rows:
    if len(r) != 4:
      raise ValueError("bad target row %r" % (r,))
  idx = np.array([int(r[0]) for r in rows], np.int64)
  delta = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
  if idx.min() < 0 or idx.max() >= nverts:
    raise ValueError("target vertex index out of range")
  # Keep the LAST occurrence of a repeated index.
  rev_unique, rev_pos = np.unique(idx[::-1], return_index=True)
  keep = len(idx) - 1 - rev_pos
  return idx[keep], delta[keep]


def _load_gz_target(path, nverts):
  with gzip.open(path, "rt") as f:
    return _parse_target_text(f.read(), nverts)


def _load_plain_target(path, nverts):
  with open(path, "r") as f:
    return _parse_target_text(f.read(), nverts)


class _Shapes(object):
  """Blend shapes accumulated as sparse (idx, delta) pairs + labels."""

  def __init__(self, nverts):
    self.nverts = nverts
    self.idx, self.delta, self.labels = [], [], []

  def add(self, label, idx, delta):
    self.labels.append(label)
    self.idx.append(idx)
    self.delta.append(delta)

  def dense(self, i):
    out = np.zeros((self.nverts, 3))
    out[self.idx[i]] = self.delta[i]
    return out


def _macro_shapes(root, template, world, shapes):
  """load_macrodetails + stacking order of load_all_blendshapes."""
  pv = PHENOTYPE_VARIATIONS
  nv = len(template)
  md = os.path.join(root, "mpfb2", "targets", "macrodetails")
  newborn_scale = np.array([0.922, 0.922, 0.75])
  norm = 3.0

  def load(path, newborn):
    idx, delta = _load_gz_target(path, nv)
    delta = _apply(world, delta)
    if not newborn:
      return idx, delta
    dense = np.zeros((nv, 3))
    dense[idx] = delta
    dense = newborn_scale[None] * dense + ((newborn_scale[None] - 1) / norm) * template
    return np.arange(nv), dense

  def a2l(age):
    return age if age != "newborn" else "baby"

  masks = []
  all_keys = [z for vals in pv.values() for z in vals]

  def mask_of(components):
    m = np.zeros(len(all_keys))
    for x in components:
      m[all_keys.index(x)] = 1
    return m

  blocks = []
  universal = []
  for g in pv["gender"]:
    for a in pv["age"]:
      for mu in pv["muscle"]:
        for w in pv["weight"]:
          p = os.path.join(md, "universal-%s-%s-%s-%s.target.gz"
                           % (g, a2l(a), mu, w))
          universal.append(((g, a, mu, w), load(p, a == "newborn")))
  blocks.append(("universal", universal))
  race = []
  for r in pv["race"]:
    for g in pv["gender"]:
      for a in pv["age"]:
        p = os.path.join(md, "%s-%s-%s.target.gz" % (r, g, a2l(a)))
        race.append(((r, g, a), load(p, a == "newborn")))
  blocks.append(("race", race))
  height = []
  for g in pv["gender"]:
    for a in pv["age"]:
      for mu in pv["muscle"]:
        for w in pv["weight"]:
          for h in pv["height"]:
            p = os.path.join(md, "height", "%s-%s-%s-%s-%s.target.gz"
                             % (g, a2l(a), mu, w, h))
            height.append(((g, a, mu, w, h), load(p, a == "newborn")))
  blocks.append(("height", height))
  props = []
  for g in pv["gender"]:
    for a in pv["age"]:
      if a in ("newborn", "baby"):
        continue
      for mu in pv["muscle"]:
        for w in pv["weight"]:
          for pr in pv["proportions"]:
            p = os.path.join(md, "proportions", "%s-%s-%s-%s-%s.target.gz"
                             % (g, a, mu, w, pr))
            props.append(((g, a, mu, w, pr), load(p, False)))
  blocks.append(("proportions", props))
  breast = []
  bd = os.path.join(root, "mpfb2", "targets", "breast")
  g = "female"
  for a in pv["age"]:
    for mu in pv["muscle"]:
      for w in pv["weight"]:
        for c in pv["cupsize"]:
          for fi in pv["firmness"]:
            p = os.path.join(bd, "%s-%s-%s-%s-%s-%s.target.gz"
                             % (g, a, mu, w, c, fi))
            if os.path.exists(p):
              breast.append(((g, a, mu, w, c, fi), load(p, False)))
  blocks.append(("breast", breast))

  for block, items in blocks:
    for comps, (idx, delta) in items:
      shapes.add("%s:%s" % (block, "-".join(comps)), idx, delta)
      masks.append(mask_of(comps))
  return np.stack(masks)


def _local_change_shapes(root, nv, world, shapes):
  """Opposite target pairs from target.json, stacked (pos, neg)."""
  with open(os.path.join(root, "mpfb2", "targets", "target.json")) as f:
    meta = json.load(f)
  labels = []
  regions = []
  for key, m in meta.items():
    if key == "genitals":
      continue
    for cat in m["categories"]:
      for side in ("left", "right", "unsided"):
        if "opposites" not in cat:
          continue
        neg = cat["opposites"]["negative-%s" % side]
        pos = cat["opposites"]["positive-%s" % side]
        if not (neg and pos):
          continue
        tdir = os.path.join(root, "mpfb2", "targets", key)
        n_idx, n_delta = _load_gz_target(os.path.join(tdir, neg + ".target.gz"), nv)
        p_idx, p_delta = _load_gz_target(os.path.join(tdir, pos + ".target.gz"), nv)
        shapes.add("local_change:%s" % pos, p_idx, _apply(world, p_delta))
        shapes.add("local_change:%s" % neg, n_idx, _apply(world, n_delta))
        labels.append(pos)
        regions.append(key)
  return labels, regions


def _load_rig(rig_path, weights_path, nverts):
  with open(rig_path) as f:
    rig = json.load(f)
  if "bones" in rig:
    rig = rig["bones"]
  roots = [n for n in rig if ("parent" not in rig[n] or rig[n]["parent"] == "")]
  assert len(roots) == 1
  labels, parents = [], []

  def walk(label, parent):
    bid = len(labels)
    labels.append(label)
    parents.append(parent)
    for node in rig.keys():
      if node not in labels and rig[node]["parent"] == label:
        walk(node, bid)

  walk(roots[0], -1)
  assert len(labels) == len(rig)
  with open(weights_path) as f:
    wdata = json.load(f)["weights"]
  dense = np.zeros((nverts, len(labels)))
  for bid, lab in enumerate(labels):
    for vidx, w in sorted(wdata.get(lab, [])):
      dense[vidx, bid] += w
  with np.errstate(invalid="ignore", divide="ignore"):
    dense = dense / dense.sum(axis=1, keepdims=True)
  return labels, parents, dense


def _filter_rig(labels, parents, dense, to_remove):
  """model_data._filter_rig with subtree_root=None (weights to ancestors)."""
  keep = [i for i, l in enumerate(labels) if l not in to_remove]
  keep_set = set(keep)
  old_to_new = {o: n for n, o in enumerate(keep)}

  def target(i):
    while i >= 0:
      if i in keep_set:
        return old_to_new[i]
      i = parents[i]
    raise ValueError("bone without retained ancestor")

  new_dense = np.zeros((dense.shape[0], len(keep)))
  for i in range(len(labels)):
    new_dense[:, target(i)] += dense[:, i]
  new_parents = []
  for o in keep:
    p = parents[o]
    while p >= 0 and p not in keep_set:
      p = parents[p]
    new_parents.append(-1 if p < 0 else old_to_new[p])
  return [labels[o] for o in keep], new_parents, new_dense


def build_model(data_dir):
  """Build the model arrays from an extracted ``src/anny/data`` folder.

  Returns a dict of numpy arrays/lists (see ``save_compiled``)."""
  world = _world_matrix()
  v, uvs, groups = load_obj(os.path.join(data_dir, "mpfb2", "3dobjs", "base.obj"))
  template = _apply(world, v)
  nv = len(template)

  # faces: body + eyes + tongue (topology "anny": eyes=True, tongue=True)
  parts = [("body", "body"), ("helper-l-eye", "eye_l"),
           ("helper-r-eye", "eye_r"), ("helper-tongue", "tongue")]
  faces = np.concatenate([groups[g][0] for g, _ in parts])
  ftci = np.concatenate([groups[g][1] for g, _ in parts])
  part_names = [p for _, p in parts]
  face_part = np.concatenate([np.full(len(groups[g][0]), k, np.int32)
                              for k, (g, _) in enumerate(parts)])

  shapes = _Shapes(nv)
  macro_mask = _macro_shapes(data_dir, template, world, shapes)
  n_macro = len(shapes.labels)
  fdir = os.path.join(data_dir, "faceunits01", "targets", "faceunits")
  for lab in FACIAL_ACTION_LABELS:
    idx, delta = _load_plain_target(os.path.join(fdir, lab + ".target"), nv)
    shapes.add("facial_action:%s" % lab, idx, _apply(world, delta))
  local_labels, local_regions = _local_change_shapes(data_dir, nv, world, shapes)

  # local_changes="default": drop nipple targets.
  keep_rows = [i for i, l in enumerate(shapes.labels)
               if not (l.startswith("local_change:") and "nipple" in l.lower())]
  kept_local = [(l, r) for l, r in zip(local_labels, local_regions)
                if "nipple" not in l.lower()]
  assert len(keep_rows) == n_macro + 52 + 2 * len(kept_local)

  # rig
  rig_dir = os.path.join(data_dir, "mpfb2", "rigs", "standard")
  b_labels, b_parents, weights = _load_rig(
      os.path.join(rig_dir, "rig.default.json"),
      os.path.join(rig_dir, "weights.default.json"), nv)
  b_labels, b_parents, weights = _filter_rig(b_labels, b_parents, weights,
                                             ANNY_BONES_TO_REMOVE)

  # nudity edit
  drop = np.isin(faces, _DISCARD_VERTS).any(axis=1)
  vt_of = {}
  for fi in np.nonzero(drop)[0]:
    for vid, tid in zip(faces[fi], ftci[fi]):
      vt_of.setdefault(int(vid), int(tid))
  cap_l = np.array(_CAP_L)
  cap_r = np.array(_CAP_R)
  t_l = np.array([[vt_of[int(x)] for x in row] for row in cap_l])
  t_r = np.array([[vt_of[int(x)] for x in row] for row in cap_r])
  faces = np.concatenate([faces[~drop], cap_l, cap_r])
  ftci = np.concatenate([ftci[~drop], t_l, t_r])
  face_part = np.concatenate([face_part[~drop],
                              np.zeros(len(cap_l) + len(cap_r), np.int32)])

  # remove unattached vertices
  used = np.unique(faces.ravel())
  old_to_new = np.full(nv, -1, np.int64)
  old_to_new[used] = np.arange(len(used))
  faces = old_to_new[faces]
  template = template[used]
  weights = weights[used]

  # sparse blend shapes restricted to the kept vertices
  ptr = [0]
  all_idx, all_delta, labels = [], [], []
  for i in keep_rows:
    idx = old_to_new[shapes.idx[i]]
    ok = idx >= 0
    all_idx.append(idx[ok].astype(np.int32))
    all_delta.append(shapes.delta[i][ok])
    ptr.append(ptr[-1] + int(ok.sum()))
    labels.append(shapes.labels[i])

  # cached procrustes orientation (anny.pth), selected by labels
  cache = load_pth(os.path.join(data_dir, "cached", "anny.pth"))
  src = {l: i for i, l in enumerate(cache["bone_labels"])}
  sel = [src[l] for l in b_labels]
  src_bs = {l: i for i, l in enumerate(cache["blendshape_labels"])}
  missing = [l for l in labels if l not in src_bs
             and not l.startswith("facial_action:")]
  if missing:
    raise ValueError("anny.pth lacks blend shapes: %s" % missing[:5])
  rows = [src_bs.get(l, -1) for l in labels]

  def rows_of(arr):
    arr = arr[:, sel]
    pad = np.concatenate([arr, np.zeros((1,) + arr.shape[1:])])
    return pad[rows]

  # Order in which local changes are exposed (pos labels) + their region.
  return dict(
      version=COMPILED_VERSION,
      template=template,
      faces=faces.astype(np.int32),
      uvs=uvs,
      face_uv=ftci.astype(np.int32),
      face_part=face_part,
      part_names=part_names,
      base_vertex_indices=used.astype(np.int32),
      bs_ptr=np.asarray(ptr, np.int64),
      bs_idx=np.concatenate(all_idx),
      bs_delta=np.concatenate(all_delta),
      bs_labels=labels,
      macro_mask=macro_mask,
      local_labels=[l for l, _ in kept_local],
      local_regions=[r for _, r in kept_local],
      bone_labels=b_labels,
      bone_parents=np.asarray(b_parents, np.int32),
      weights=weights,
      heads=cache["template_bone_heads"][sel],
      heads_bs=rows_of(cache["bone_heads_blendshapes"]),
      orient=cache["bone_template_orientation_matrices"][sel],
      orient_bs=rows_of(cache["bone_orientation_blendshapes"]),
      ref_orient=cache["reference_bone_orientations"][sel],
  )


_STR_LISTS = ("part_names", "bs_labels", "local_labels", "local_regions",
              "bone_labels")


def save_compiled(arrays, path):
  out = {}
  for k, v in arrays.items():
    if k in _STR_LISTS:
      out[k] = np.asarray(json.dumps(list(v)))
    else:
      out[k] = np.asarray(v)
  tmp = path + ".tmp.npz"
  np.savez_compressed(tmp, **out)
  os.replace(tmp, path)


def load_compiled(path):
  with np.load(path, allow_pickle=False) as z:
    d = {k: z[k] for k in z.files}
  for k in _STR_LISTS:
    d[k] = json.loads(str(d[k]))
  if int(d["version"]) != COMPILED_VERSION:
    raise ValueError("stale compiled model")
  return d


# --- math helpers -----------------------------------------------------------

def interpolation_coefficients(value, anchors, extrapolate=False):
  """anny.utils.interpolation.linear_interpolation_coefficients (scalar)."""
  anchors = np.asarray(anchors, np.float64)
  n = len(anchors)
  v = float(value)
  if not extrapolate:
    v = min(max(v, anchors[0]), anchors[-1])
  idx = int(np.searchsorted(anchors, v, side="left"))
  idx = min(max(idx, 1), n - 1)
  lo, hi = anchors[idx - 1], anchors[idx]
  alpha = (v - lo) / (hi - lo)
  w = np.zeros(n)
  w[idx - 1] = 1.0 - alpha
  w[idx] = alpha
  return w


def special_procrustes(m):
  """Batched closest rotation to each 3x3 (roma.special_procrustes)."""
  u, _s, vt = np.linalg.svd(m)
  d = np.sign(np.linalg.det(u @ vt))
  u = u.copy()
  u[..., :, -1] *= d[..., None]
  return u @ vt


def rigid_inverse(h):
  out = np.zeros_like(h)
  r = h[..., :3, :3]
  rt = np.swapaxes(r, -1, -2)
  out[..., :3, :3] = rt
  out[..., :3, 3] = -np.einsum("...ij,...j->...i", rt, h[..., :3, 3])
  out[..., 3, 3] = 1.0
  return out


def euler_xyz_to_matrix(rx, ry, rz):
  """Rotation applying X, then Y, then Z (Maya's 'xyz' rotate order)."""
  cx, sx = np.cos(rx), np.sin(rx)
  cy, sy = np.cos(ry), np.sin(ry)
  cz, sz = np.cos(rz), np.sin(rz)
  mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  return mz @ my @ mx


# --- the model --------------------------------------------------------------

class AnnyModel(object):
  """Evaluates the compiled Anny arrays (see ``build_model``)."""

  def __init__(self, arrays, extrapolate_phenotypes=False):
    a = arrays
    self.template = a["template"]
    self.faces = a["faces"]
    self.uvs = a["uvs"]
    self.face_uv = a["face_uv"]
    self.face_part = a["face_part"]
    self.part_names = list(a["part_names"])
    self.base_vertex_indices = a["base_vertex_indices"]
    self.bs_ptr = a["bs_ptr"]
    self.bs_idx = a["bs_idx"]
    self.bs_delta = a["bs_delta"]
    self.bs_labels = list(a["bs_labels"])
    self.macro_mask = a["macro_mask"]
    self.local_labels = list(a["local_labels"])
    self.local_regions = list(a["local_regions"])
    self.bone_labels = list(a["bone_labels"])
    self.bone_parents = [int(p) for p in a["bone_parents"]]
    self.weights = a["weights"]
    self.heads = a["heads"]
    self.heads_bs = a["heads_bs"]
    self.orient = a["orient"]
    self.orient_bs = a["orient_bs"]
    self.ref_orient = a["ref_orient"]
    self.extrapolate = extrapolate_phenotypes

    self.num_vertices = len(self.template)
    self.num_bones = len(self.bone_labels)
    self.num_shapes = len(self.bs_labels)
    self.n_macro = len(self.macro_mask)
    self.face_offset = self.n_macro
    self.local_offset = self.n_macro + len(FACIAL_ACTION_LABELS)
    assert self.num_shapes == self.local_offset + 2 * len(self.local_labels)
    for i, p in enumerate(self.bone_parents):
      assert p < i, "bones must be topologically sorted"
    self._heads_bs_flat = self.heads_bs.reshape(self.num_shapes, -1)
    self._orient_bs_flat = self.orient_bs.reshape(self.num_shapes, -1)
    # Sparse skinning: per bone the vertex rows it influences.
    self._anchors = {
        "age": np.linspace(-1.0 / 3.0, 1.0, 5),
    }
    for k in ("gender", "muscle", "weight", "height", "proportions",
              "cupsize", "firmness"):
      self._anchors[k] = np.linspace(0.0, 1.0, len(PHENOTYPE_VARIATIONS[k]))
    self._local_index = {l: i for i, l in enumerate(self.local_labels)}
    self._face_index = {l: i for i, l in enumerate(FACIAL_ACTION_LABELS)}

  # -- coefficients ----------------------------------------------------------

  def macro_weights(self, phenotypes=None):
    """Weights of the stacked macro blend shapes for a phenotype dict
    (missing keys default to 0.5, as in Anny)."""
    p = dict(phenotypes or {})
    unknown = set(p) - set(PHENOTYPE_LABELS)
    if unknown:
      raise ValueError("unknown phenotypes: %s" % sorted(unknown))
    val = lambda k: float(p.get(k, 0.5))
    feats = {}
    for f in ("age", "gender", "muscle", "weight", "height", "proportions",
              "cupsize", "firmness"):
      w = interpolation_coefficients(val(f), self._anchors[f], self.extrapolate)
      feats.update({k: w[i] for i, k in enumerate(PHENOTYPE_VARIATIONS[f])})
    race = np.array([val("african"), val("asian"), val("caucasian")])
    s = race.sum()
    race = race / s if s != 0 and np.isfinite(race / s).all() else np.full(3, 1 / 3)
    feats.update(african=race[0], asian=race[1], caucasian=race[2])
    phens = np.array([feats[k] for vals in PHENOTYPE_VARIATIONS.values()
                      for k in vals])
    m = self.macro_mask
    return np.prod(phens[None] * m + (1 - m), axis=-1)

  def coefficients(self, phenotypes=None, local_changes=None,
                   facial_actions=None):
    c = np.zeros(self.num_shapes)
    c[:self.n_macro] = self.macro_weights(phenotypes)
    for k, v in (facial_actions or {}).items():
      c[self.face_offset + self._face_index[k]] = float(v)
    for k, v in (local_changes or {}).items():
      i = self._local_index[k]
      v = float(v)
      c[self.local_offset + 2 * i] = v if v >= 0 else 0.0
      c[self.local_offset + 2 * i + 1] = -v if v <= 0 else 0.0
    return c

  # -- rest model ------------------------------------------------------------

  def rest_vertices(self, coeffs):
    out = self.template.copy()
    for i in np.nonzero(coeffs)[0]:
      a, b = self.bs_ptr[i], self.bs_ptr[i + 1]
      if b > a:
        out[self.bs_idx[a:b]] += coeffs[i] * self.bs_delta[a:b]
    return out

  def blend_vertices(self, coeffs, indices):
    """Sum of coeff-weighted deltas only (no template), e.g. one target."""
    out = np.zeros_like(self.template)
    for i in indices:
      a, b = self.bs_ptr[i], self.bs_ptr[i + 1]
      out[self.bs_idx[a:b]] += coeffs[i] * self.bs_delta[a:b]
    return out

  def rest_skeleton(self, coeffs):
    heads = self.heads + (coeffs @ self._heads_bs_flat).reshape(-1, 3)
    m = self.orient + (coeffs @ self._orient_bs_flat).reshape(-1, 3, 3)
    poses = np.zeros((self.num_bones, 4, 4))
    poses[:, :3, :3] = special_procrustes(m)
    poses[:, :3, 3] = heads
    poses[:, 3, 3] = 1.0
    return poses  # root_identity_orientation is False for the anny rig

  # -- kinematics ------------------------------------------------------------

  def reference_poses(self, rest_poses):
    """FK with absolute reference orientations (the 'neutral' pose)."""
    n = self.num_bones
    poses = np.zeros_like(rest_poses)
    transforms = np.zeros_like(rest_poses)
    rest_inv = rigid_inverse(rest_poses)
    for i in range(n):
      p = self.bone_parents[i]
      if p < 0:
        # As in kinematics.parallel_forward_kinematic_absolute_orientations:
        # a root's transform is computed BEFORE its orientation override.
        pose = rest_poses[i].copy()
        transforms[i] = pose @ rest_inv[i]
        pose[:3, :3] = self.ref_orient[i]
      else:
        pose = transforms[p] @ rest_poses[i]
        pose[:3, :3] = self.ref_orient[i]
        transforms[i] = pose @ rest_inv[i]
      poses[i] = pose
    return poses

  def posed_bones(self, rest_poses, deltas=None):
    """local-ref FK. ``deltas``: (B,4,4) per-bone transforms expressed in
    the reference (world-aligned) frame; None = identity (neutral pose).
    Returns (bone_poses, bone_transforms)."""
    n = self.num_bones
    if deltas is None:
      deltas = np.broadcast_to(np.eye(4), (n, 4, 4))
    ref = self.reference_poses(rest_poses)
    rref = np.zeros((n, 4, 4))
    rref[:, :3, :3] = ref[:, :3, :3]
    rref[:, 3, 3] = 1.0
    t = rigid_inverse(rref) @ deltas @ rref
    tt = ref @ t
    ref_inv = rigid_inverse(ref)
    base = ref_inv[0]
    poses = np.zeros_like(rest_poses)
    trans = np.zeros_like(rest_poses)
    for i in range(n):
      p = self.bone_parents[i]
      poses[i] = (base @ tt[i]) if p < 0 else trans[p] @ tt[i]
      trans[i] = poses[i] @ ref_inv[i]
    bone_transforms = poses @ rigid_inverse(rest_poses)
    return poses, bone_transforms

  def skin(self, rest_vertices, bone_transforms):
    blended = self.weights @ bone_transforms[:, :3, :].reshape(self.num_bones, 12)
    blended = blended.reshape(-1, 3, 4)
    return (np.einsum("vij,vj->vi", blended[:, :, :3], rest_vertices)
            + blended[:, :, 3])

  def evaluate(self, coeffs, deltas=None):
    rest_v = self.rest_vertices(coeffs)
    rest_p = self.rest_skeleton(coeffs)
    poses, transforms = self.posed_bones(rest_p, deltas)
    verts = self.skin(rest_v, transforms)
    return dict(rest_vertices=rest_v, rest_bone_poses=rest_p,
                bone_poses=poses, bone_transforms=transforms, vertices=verts)

  # -- conveniences ----------------------------------------------------------

  def ground_offset(self, rest_vertices, rest_poses):
    """Height (m) lifting the neutral pose so the soles touch Z=0: the posed
    root sits at the origin, so lift by the rest root height above the
    lowest rest vertex. Shape-dependent only (stable while posing)."""
    return float(rest_poses[0, 2, 3] - rest_vertices[:, 2].min())


def deltas_from_euler(model, rotations_deg, root_translation=None):
  """Per-bone local-ref deltas from Maya-axes XYZ Euler degrees.

  ``rotations_deg``: {bone_label: [rx, ry, rz]} about Maya world axes (Y up)
  at the neutral pose; ``root_translation``: Maya cm. Converted into Anny's
  frame (rotation conjugated by the basis change)."""
  n = model.num_bones
  d = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
  c = MAYA_BASIS
  index = {l: i for i, l in enumerate(model.bone_labels)}
  for lab, r in (rotations_deg or {}).items():
    if lab not in index or not any(abs(float(x)) > 0 for x in r):
      continue
    rm = euler_xyz_to_matrix(*np.deg2rad([float(x) for x in r]))
    d[index[lab], :3, :3] = c.T @ rm @ c
  if root_translation is not None:
    d[0, :3, 3] = c.T @ (np.asarray(root_translation, np.float64) / MAYA_SCALE)
  return d


def to_maya(points, lift=0.0):
  """(N,3) Anny metres Z-up -> Maya cm Y-up, lifted by ``lift`` metres."""
  p = np.asarray(points, np.float64) @ MAYA_BASIS.T * MAYA_SCALE
  p[:, 1] += lift * MAYA_SCALE
  return p


def matrices_to_maya(poses, lift=0.0):
  """(B,4,4) Anny bone frames -> Maya world matrices (rotation C*R, cm)."""
  out = np.zeros_like(poses)
  out[:, :3, :3] = MAYA_BASIS @ poses[:, :3, :3]
  out[:, :3, 3] = to_maya(poses[:, :3, 3], lift)
  out[:, 3, 3] = 1.0
  return out


def ensure_compiled(data_dir, cache_dir=None, log=None):
  """Load the compiled arrays, (re)building them from source if missing."""
  cache_dir = cache_dir or data_dir
  path = os.path.join(cache_dir, COMPILED_NAME)
  if os.path.isfile(path):
    try:
      return load_compiled(path)
    except Exception:
      pass
  if log:
    log("Compiling Anny model from source assets (one time)...")
  arrays = build_model(data_dir)
  save_compiled(arrays, path)
  return load_compiled(path)
