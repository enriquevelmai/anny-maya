#!/usr/bin/env python
"""Persistent Anny worker: load the model once, serve body evals on stdin.

Protocol (line-oriented, one JSON object per line):

  startup   -> writes topology into --session dir, prints:  READY <session>
  eval      <- {"phenotypes": {...}, "local_changes": {...},
                "facial_actions": {...}, "rotations": {bone: [rx,ry,rz]},
                "translation": [x,y,z], "out": "<path/verts.bin>"}
            -> OK <out>       (float32 V x 3, Maya cm, Y up, feet on Y=0)
  bake      <- same state + {"cmd": "bake"}
            -> BAKED <n>      (rig.json + rig_*.bin in the session dir)
  pose      <- {"cmd": "pose_preset", "name": "tpose", ...state}
            -> POSE {bone: [rx,ry,rz], ...}
  quit      <- {"cmd": "quit"}  -> BYE
  error     -> ERR <message>    (the server stays alive)

Missing params default to Anny's neutral values (phenotypes 0.5, the rest 0).
Rotations are XYZ Euler degrees about Maya's world axes at the neutral pose
(Anny's ``local-ref`` pose parameterization); translation is Maya cm.

    python server.py --compile        # one-time: build the compiled model

Runs inside the module's own runtime (numpy only), driven by
anny_maya.core.worker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import _anny_core as core  # noqa: E402
import _head  # noqa: E402

DATA_ROOT = os.path.join(HERE, "anny_data")
DATA_DIR = os.path.join(DATA_ROOT, "src", "anny", "data")
GNM_ROOT = os.path.join(HERE, "gnm_data")
GNM_NPZ = os.path.join(GNM_ROOT, _head.GNM_NPZ_REL)

# Anatomical grouping of the 52 ARKit face units (Face tab sections).
FACE_GROUPS = [
    ("Brows", ["browDownLeft", "browDownRight", "browInnerUp",
               "browOuterUpLeft", "browOuterUpRight"]),
    ("Eyes", ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
              "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
              "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
              "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
              "eyeWideLeft", "eyeWideRight"]),
    ("Cheeks & Nose", ["cheekPuff", "cheekSquintLeft", "cheekSquintRight",
                       "noseSneerLeft", "noseSneerRight"]),
    ("Jaw", ["jawForward", "jawLeft", "jawOpen", "jawRight"]),
    ("Mouth", ["mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
               "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
               "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
               "mouthPressLeft", "mouthPressRight", "mouthPucker",
               "mouthRight", "mouthRollLower", "mouthRollUpper",
               "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
               "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
               "mouthUpperUpLeft", "mouthUpperUpRight"]),
    ("Tongue", ["tongueOut"]),
]

# Bone groups for the Pose tab (labels as in the anny rig).
def _bone_groups(labels):
  def pick(pred):
    return [l for l in labels if pred(l)]
  hand = ("finger", "metacarpal")
  return [
      ("Spine & Head", pick(lambda l: l == "root" or l.startswith("spine")
                            or l.startswith("neck") or l in ("head", "eye.L",
                                                             "eye.R"))),
      ("Left Arm", pick(lambda l: l.endswith(".L") and any(
          l.startswith(p) for p in ("clavicle", "shoulder", "upperarm",
                                    "lowerarm", "wrist")))),
      ("Right Arm", pick(lambda l: l.endswith(".R") and any(
          l.startswith(p) for p in ("clavicle", "shoulder", "upperarm",
                                    "lowerarm", "wrist")))),
      ("Left Hand", pick(lambda l: l.endswith(".L") and l.startswith(hand))),
      ("Right Hand", pick(lambda l: l.endswith(".R") and l.startswith(hand))),
      ("Left Leg", pick(lambda l: l.endswith(".L") and any(
          l.startswith(p) for p in ("pelvis", "upperleg", "lowerleg",
                                    "foot")))),
      ("Right Leg", pick(lambda l: l.endswith(".R") and any(
          l.startswith(p) for p in ("pelvis", "upperleg", "lowerleg",
                                    "foot")))),
      ("Left Toes", pick(lambda l: l.endswith(".L") and l.startswith("toe"))),
      ("Right Toes", pick(lambda l: l.endswith(".R") and l.startswith("toe"))),
  ]


def _log(*a):
  print(*a, file=sys.stderr, flush=True)


def _mirror_pairs(names, left, right, where="suffix"):
  idx = {n: i for i, n in enumerate(names)}
  pairs = []
  for n in names:
    if where == "suffix" and n.endswith(left):
      m = n[: -len(left)] + right
    elif where == "prefix" and n.startswith(left):
      m = right + n[len(left):]
    else:
      continue
    if m in idx:
      pairs.append([n, m])
  return pairs


def export_topology(model, session, head=None):
  """Constant topology files for Maya (pure-python readers on that side)."""
  faces = np.asarray(model.faces, "<i4")
  faces.reshape(-1).tofile(os.path.join(session, "faces.bin"))
  np.full(len(faces), faces.shape[1], "<i4").tofile(
      os.path.join(session, "face_counts.bin"))
  uv = model.uvs[model.face_uv.reshape(-1)].astype("<f4")
  uv.reshape(-1).tofile(os.path.join(session, "face_uvs.bin"))
  np.asarray(model.face_part, "<i4").tofile(os.path.join(session,
                                                         "face_parts.bin"))
  neg_labels = [model.bs_labels[model.local_offset + 2 * i + 1].split(":")[1]
                for i in range(len(model.local_labels))]
  meta = {
      "num_vertices": int(model.num_vertices),
      "num_faces": int(len(faces)),
      "verts_per_face": int(faces.shape[1]),
      "part_names": model.part_names,
      "bone_labels": model.bone_labels,
      "bone_parents": model.bone_parents,
      "bone_groups": _bone_groups(model.bone_labels),
      "bone_mirror": _mirror_pairs(model.bone_labels, ".L", ".R"),
      "phenotype_labels": core.PHENOTYPE_LABELS,
      "local_labels": model.local_labels,
      "local_neg_labels": neg_labels,
      "local_regions": model.local_regions,
      "local_mirror": _mirror_pairs(model.local_labels, "l-", "r-", "prefix"),
      "face_labels": core.FACIAL_ACTION_LABELS,
      "face_groups": FACE_GROUPS,
      "face_mirror": _mirror_pairs(core.FACIAL_ACTION_LABELS, "Left", "Right"),
      "anny": _anny_pin(),
      "head": export_head_topology(head, session) if head else None,
  }
  with open(os.path.join(session, "topology.json"), "w") as f:
    json.dump(meta, f)
  return meta


def export_head_topology(head, session):
  """The combined body + GNM head mesh (mixed quads/triangles)."""
  head.faces_flat.astype("<i4").tofile(os.path.join(session, "head_faces.bin"))
  head.face_counts.astype("<i4").tofile(
      os.path.join(session, "head_face_counts.bin"))
  head.face_uvs_flat.astype("<f4").reshape(-1).tofile(
      os.path.join(session, "head_face_uvs.bin"))
  head.face_part.astype("<i4").tofile(
      os.path.join(session, "head_face_parts.bin"))
  ident, expr = _head.gnm_names(GNM_NPZ)
  return {
      "num_vertices": int(head.num_vertices),
      "body_vertices": int(len(head.anny_used)),  # then the GNM vertices
      "num_faces": int(len(head.face_counts)),
      "part_names": head.part_names,
      "identity_dim": len(ident),
      "expression_dim": len(expr),
      "identity_groups": _head.group_ranges(ident),
      "expression_groups": _head.group_ranges(expr),
      "gnm": _pin(os.path.join(GNM_ROOT, "GNM_DATA.json")),
  }


def _pin(path):
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return {}


def _anny_pin():
  return _pin(os.path.join(DATA_ROOT, "ANNY_DATA.json"))


class Evaluator(object):

  def __init__(self, model, head=None):
    self.model = model
    self.head = head          # _head.HeadAttachment, or None (no GNM data)
    self._lift_key = None
    self._lift = 0.0

  def head_req(self, req):
    """The request's GNM head params, or None (Anny's own head)."""
    h = req.get("head")
    if not h:
      return None
    if self.head is None:
      raise RuntimeError("GNM head data is not installed (Head tab > Use "
                         "GNM head downloads it).")
    return h

  def head_fit(self, h):
    fit = dict(h.get("fit") or {})
    if "offset" in fit:  # Maya cm -> Anny metres
      fit["offset"] = (core.MAYA_BASIS.T @ (np.asarray(fit["offset"], float)
                                            / core.MAYA_SCALE)).tolist()
    return fit

  def coeffs(self, req):
    m = self.model
    face = {k: v for k, v in (req.get("facial_actions") or {}).items() if v}
    if req.get("head"):
      face = {}  # Anny's own face units only exist on the hidden head
    return m.coefficients(req.get("phenotypes") or {},
                          req.get("local_changes") or {}, face)

  def lift(self, coeffs, rest_v, rest_p):
    """Feet on the ground in the neutral pose, cached per body shape (face
    units don't move the soles, so they're excluded from the key)."""
    m = self.model
    key = np.concatenate([coeffs[:m.face_offset],
                          coeffs[m.local_offset:]]).tobytes()
    if key != self._lift_key:
      _poses, tr = m.posed_bones(rest_p, None)
      neutral = m.skin(rest_v, tr)
      self._lift = -float(neutral[:, 2].min())
      self._lift_key = key
    return self._lift

  def deltas(self, req):
    return core.deltas_from_euler(self.model, req.get("rotations") or {},
                                  req.get("translation"))

  def evaluate(self, req):
    m = self.model
    c = self.coeffs(req)
    rest_v = m.rest_vertices(c)
    rest_p = m.rest_skeleton(c)
    poses, tr = m.posed_bones(rest_p, self.deltas(req))
    lift = self.lift(c, rest_v, rest_p)
    h = self.head_req(req)
    if h is None:
      verts = m.skin(rest_v, tr)
    else:
      rest = self.head.rest_vertices(rest_v, rest_p, h.get("identity"),
                                     h.get("expression"), self.head_fit(h))
      verts = self.head.skin(rest, tr)
    return core.to_maya(verts, lift), poses, lift

  def _write_target(self, session, k, delta, name, value):
    """One blendShape target as sparse Maya-space deltas."""
    d = delta @ core.MAYA_BASIS.T * core.MAYA_SCALE
    nz = np.nonzero(np.abs(d).max(axis=1) > 1e-7)[0]
    nz.astype("<i4").tofile(os.path.join(session, "rig_face_%02d_idx.bin" % k))
    d[nz].astype("<f4").tofile(os.path.join(session, "rig_face_%02d.bin" % k))
    return {"name": name, "count": int(len(nz)), "value": float(value)}

  def bake(self, req, session):
    """Rig data for Maya: bind mesh (current shape, face neutral), the 52
    face targets as deltas, skin weights, rest + posed joint matrices."""
    m = self.model
    c = self.coeffs(req)
    face_vals = dict(req.get("facial_actions") or {})
    c_noface = c.copy()
    c_noface[m.face_offset:m.local_offset] = 0.0
    rest_v = m.rest_vertices(c_noface)
    rest_p = m.rest_skeleton(c_noface)
    lift = self.lift(c_noface, rest_v, rest_p)
    poses, _tr = m.posed_bones(rest_p, self.deltas(req))
    targets = []
    h = self.head_req(req)
    if h is None:
      rest, weights = rest_v, m.weights
      ones = np.ones(m.num_shapes)
      for k, lab in enumerate(core.FACIAL_ACTION_LABELS):
        targets.append(self._write_target(
            session, k, m.blend_vertices(ones, [m.face_offset + k]), lab,
            face_vals.get(lab, 0.0)))
    else:
      # Neutral GNM expression in the bind mesh; the panel's expression as
      # one target, plus the first N modes of each expression region.
      hd, fit, ident = self.head, self.head_fit(h), h.get("identity")
      expr = list(h.get("expression") or [])
      rest = hd.rest_vertices(rest_v, rest_p, ident, None, fit)
      weights = hd.weights
      if any(expr):
        cur = hd.rest_vertices(rest_v, rest_p, ident, expr, fit)
        targets.append(self._write_target(session, 0, cur - rest,
                                          "gnmExpression", 1.0))
      n = int(req.get("modes", 0))
      if n > 0:
        _ident, names = _head.gnm_names(GNM_NPZ)
        for _lab, start, end in _head.group_ranges(names):
          for i in range(start, min(end + 1, start + n)):
            e = np.zeros(len(names))
            e[i] = 2.0  # a unit target reaches +2 sigma at weight 1
            mode = hd.rest_vertices(rest_v, rest_p, ident, e, fit)
            targets.append(self._write_target(session, len(targets),
                                              mode - rest, names[i], 0.0))
    core.to_maya(rest, lift).astype("<f4").tofile(
        os.path.join(session, "rig_rest.bin"))
    weights.astype("<f8").tofile(os.path.join(session, "rig_weights.bin"))

    def maya_rows(mats):  # Maya's row-vector convention: transpose, flatten
      mm = core.matrices_to_maya(mats, lift)
      return [[float(x) for x in a.T.reshape(-1)] for a in mm]

    meta = {
        "num_vertices": int(len(rest)),
        "head": h is not None,
        "bone_labels": m.bone_labels,
        "bone_parents": m.bone_parents,
        "rest_world": maya_rows(rest_p),
        "posed_world": maya_rows(poses),
        "targets": targets,
        "lift_cm": lift * core.MAYA_SCALE,
    }
    with open(os.path.join(session, "rig.json"), "w") as f:
      json.dump(meta, f)
    return len(targets)

  def pose_preset(self, req):
    """Arm-aiming presets: rotations (Maya Euler deg) for T-pose etc."""
    name = req.get("name", "neutral")
    m = self.model
    if name == "neutral":
      return {}
    c = self.coeffs(req)
    rest_p = m.rest_skeleton(c)
    ref, _tr = m.posed_bones(rest_p, None)
    pos = ref[:, :3, 3]
    lab = {l: i for i, l in enumerate(m.bone_labels)}
    down = np.deg2rad(req.get("angle", 75.0 if name == "arms_down" else 0.0))
    chains = []
    for side, sx in (("L", 1.0), ("R", -1.0)):
      t = np.array([sx * np.cos(down), 0.0, -np.sin(down)])
      chains += [("upperarm01.%s" % side, "lowerarm01.%s" % side, t),
                 ("lowerarm01.%s" % side, "wrist.%s" % side, t)]
      if name == "tpose":
        chains.append(("wrist.%s" % side, "finger3-1.%s" % side, t))
    deltas = {}

    def world_rot(b):
      r = np.eye(3)
      chain = []
      while b >= 0:
        chain.append(b)
        b = m.bone_parents[b]
      for x in reversed(chain):
        r = r @ deltas.get(x, np.eye(3))
      return r

    for bone, child, target in chains:
      b, cb = lab[bone], lab[child]
      v = pos[cb] - pos[b]
      dp = world_rot(m.bone_parents[b])
      deltas[b] = _rot_between(v, dp.T @ target)
    out = {}
    for b, d in deltas.items():
      rm = core.MAYA_BASIS @ d @ core.MAYA_BASIS.T
      out[m.bone_labels[b]] = [float(x) for x in np.rad2deg(_euler_xyz(rm))]
    return out


def _rot_between(a, b):
  a = a / np.linalg.norm(a)
  b = b / np.linalg.norm(b)
  v = np.cross(a, b)
  s = np.linalg.norm(v)
  cth = float(np.dot(a, b))
  if s < 1e-12:
    return np.eye(3) if cth > 0 else -np.eye(3)
  k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + k + k @ k * ((1 - cth) / (s * s))


def _euler_xyz(r):
  """Inverse of core.euler_xyz_to_matrix (R = Rz Ry Rx)."""
  y = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
  x = np.arctan2(r[2, 1], r[2, 2])
  z = np.arctan2(r[1, 0], r[0, 0])
  return np.array([x, y, z])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--session", help="Dir for topology + meshes.")
  p.add_argument("--data", default=DATA_DIR)
  p.add_argument("--cache", default=DATA_ROOT)
  p.add_argument("--compile", action="store_true",
                 help="Build the compiled model, then exit.")
  args = p.parse_args()

  _log("Loading Anny...")
  arrays = core.ensure_compiled(args.data, args.cache, log=_log)
  if args.compile:
    print("COMPILED", flush=True)
    return
  if not args.session:
    p.error("--session is required")
  model = core.AnnyModel(arrays)
  head = None
  if os.path.isfile(GNM_NPZ):
    try:
      head = _head.ensure_head(model, GNM_NPZ, GNM_ROOT, log=_log)
    except Exception:
      import traceback
      traceback.print_exc(file=sys.stderr)
      _log("GNM head unavailable; continuing with Anny's own head.")
  ev = Evaluator(model, head)
  export_topology(model, args.session, head)
  print("READY %s" % args.session, flush=True)

  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    try:
      req = json.loads(line)
    except ValueError as e:
      print("ERR bad json: %s" % e, flush=True)
      continue
    cmd = req.get("cmd")
    if cmd == "quit":
      print("BYE", flush=True)
      return
    try:
      if cmd == "bake":
        n = ev.bake(req, args.session)
        print("BAKED %d" % n, flush=True)
      elif cmd == "pose_preset":
        print("POSE " + json.dumps(ev.pose_preset(req)), flush=True)
      else:
        verts, _poses, _lift = ev.evaluate(req)
        verts.astype("<f4").tofile(req["out"])
        print("OK %s" % req["out"], flush=True)
    except Exception as e:  # keep the server alive on a bad request
      import traceback
      traceback.print_exc(file=sys.stderr)
      print("ERR %s" % (str(e) or e.__class__.__name__).replace("\n", " "),
            flush=True)


if __name__ == "__main__":
  main()
