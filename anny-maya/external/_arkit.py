"""The 52 ARKit facial actions on the Google GNM head (numpy + h5py).

GNM has no ARKit rig. What it does ship (pinned google/GNM commit) is:

  * a semantic expression sampler - Keras conditional-VAE decoders mapping
    (latent z, one-hot class) -> a 383-dim expression coefficient vector for
    20 named expressions (surprise, smile_wide, wink_left, pucker, ...);
  * an expression basis split by region (left eye, right eye, lower face,
    tongue, pupils), and painted vertex groups (lips, brows, orbits, cheeks,
    nose, chin, eyes, teeth, tongue).

Each ARKit shape is built from those, the gnm-maya way, never by copying
Anny's face units across (a nearest-vertex transfer tears lips/eyelids):

  1. decode the class at z = 0 (the class mean: deterministic) and, for the
     upper face, keep only one eye region's coefficients (left/right split
     for free, the basis is region-decomposed);
  2. delta = coefficients @ basis, multiplied by a smooth region mask - the
     painted groups diffused over the mesh - so e.g. browInnerUp moves only
     the brows although 'surprise' also drops the jaw; symmetric shapes are
     split Left/Right with a soft midline mask;
  3. what GNM has no class for is derived: a jaw field (how much each vertex
     follows the chin when the mouth opens) splits lips upper/lower and
     drives jawLeft/Right/Forward; eye looks rotate the eyeballs.

All deltas live in GNM's frame (metres, Y up, +Z front, +X = the subject's
left, like Anny) and are computed once into ``arkit_compiled_vN.npz``; at
evaluation they are a weighted sum added to GNM's bind pose (identity
independent - GNM's expression term is additive), so slider = bake.

h5py is only needed for that one-time compile.
"""

from __future__ import annotations

import os
import re

import numpy as np

ARKIT_COMPILED_VERSION = 1
ARKIT_COMPILED_NAME = "arkit_compiled_v%d.npz" % ARKIT_COMPILED_VERSION
DECODER_REL = os.path.join("gnm", "shape", "data", "semantic_sampler",
                           "expression_decoder_model.h5")

# Class order of GNM's expression decoder (gnm/shape/semantic_sampler.py).
EXPRESSION = [
    "surprise", "disgust", "suck", "compress_face", "stretch_face", "happy",
    "squint", "platysma", "blow", "funneler", "smile_wide", "corners_down",
    "pucker", "wink_left", "wink_right", "mouth_left", "mouth_right",
    "lips_roll_in", "snarl", "tongue_center",
]
LATENT_DIM = 64

# Eye looks: eyeball rotation (degrees) at weight 1.
LOOK_DEG = {"Up": 15.0, "Down": 20.0, "In": 22.0, "Out": 22.0}
PROJECT_LAMBDA = 1e-3


# --- the decoder (numpy forward pass over the Keras weights) -------------------

def load_decoder(path):
  """Dense layers (kernel, bias) in forward order, read with h5py."""
  import h5py
  layers = []
  with h5py.File(path, "r") as f:
    mw = f["model_weights"]
    for name in mw:
      found = {}

      def visit(n, obj, found=found):
        if hasattr(obj, "shape"):
          if n.endswith("kernel:0"):
            found["k"] = np.asarray(obj[()], np.float32)
          elif n.endswith("bias:0"):
            found["b"] = np.asarray(obj[()], np.float32)

      mw[name].visititems(visit)
      if "k" in found:
        order = int(re.findall(r"\d+", name)[-1]) if re.findall(
            r"\d+", name) else 0
        layers.append((order, found["k"], found["b"]))
  layers.sort(key=lambda t: t[0])
  return [(k, b) for _, k, b in layers]


def decode(layers, name, z=None):
  """Expression coefficients for one semantic class (z = 0: the mean)."""
  oh = np.zeros((1, len(EXPRESSION)), np.float32)
  oh[0, EXPRESSION.index(name)] = 1.0
  if z is None:
    z = np.zeros((1, LATENT_DIM), np.float32)
  x = np.concatenate([z, oh], axis=-1)
  for i, (k, b) in enumerate(layers):
    x = x @ k + b
    if i < len(layers) - 1:
      x = np.maximum(x, 0.0)
  return x[0].astype(np.float64)


# --- masks ------------------------------------------------------------------------

class _Masks(object):
  """Smooth per-vertex fields on the GNM template."""

  def __init__(self, gnm):
    self.g = gnm
    self.v = gnm.template
    q = gnm.quads
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]],
                        q[:, [3, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    self.ea, self.eb = e[:, 0], e[:, 1]
    self.deg = np.bincount(np.concatenate([self.ea, self.eb]),
                           minlength=gnm.num_vertices).astype(np.float64)
    self.skin = gnm.group("skin") > 0.5

  def group(self, *names):
    out = np.zeros(self.g.num_vertices)
    for n in names:
      out = np.maximum(out, self.g.group(n))
    return out

  def smooth(self, f, iters):
    """Diffuse a field over the mesh edges (keeps the support connected:
    the lips only meet at the corners, the eyeballs never touch skin)."""
    f = np.asarray(f, np.float64).copy()
    for _ in range(int(iters)):
      s = np.zeros_like(f)
      np.add.at(s, self.ea, f[self.eb])
      np.add.at(s, self.eb, f[self.ea])
      f = (f + s) / (1.0 + self.deg)
    return f

  def region(self, names, grow=4, soften=6):
    """Painted groups, grown by ``grow`` rings then softened: 1 inside,
    smooth falloff to 0 outside."""
    f = self.group(*names)
    if grow:
      f = (self.smooth(f, grow) > 1e-3).astype(np.float64)
    f = self.smooth(f, soften)
    return np.clip(f / max(f.max(), 1e-12), 0.0, 1.0)

  def side(self, left, width=0.008):
    """Soft half-face mask across the midline (+X = subject's left)."""
    x = self.v[:, 0] * (1.0 if left else -1.0)
    s = np.clip(x / width * 0.5 + 0.5, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)

  def outer(self, x0, x1):
    """0 for |x| < x0, smooth to 1 at |x| > x1 (outer brow etc.)."""
    t = np.clip((np.abs(self.v[:, 0]) - x0) / (x1 - x0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rot(axis, deg):
  a = np.asarray(axis, np.float64)
  a = a / np.linalg.norm(a)
  t = np.deg2rad(deg)
  k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


# --- compile -----------------------------------------------------------------------

def compile_arkit(gnm, decoder_path, labels):
  """(N, V, 3) float32 deltas in GNM's frame, one per ARKit ``labels``."""
  layers = load_decoder(decoder_path)
  B = gnm.expression_basis                                  # (E, V, 3) f32
  names = gnm.expression_names
  region_idx = {}
  for i, n in enumerate(names):
    region_idx.setdefault(n.rsplit("_", 1)[0], []).append(i)
  M = _Masks(gnm)
  coef = {n: decode(layers, n) for n in EXPRESSION}

  def field(c, regions=None):
    c = np.asarray(c, np.float64)
    if regions is not None:
      keep = np.zeros(len(c), bool)
      for r in regions:
        keep[region_idx[r]] = True
      c = np.where(keep, c, 0.0)
    nz = np.nonzero(c)[0]
    return np.einsum("e,evk->vk", c[nz], B[nz].astype(np.float64))

  L, R = M.side(True), M.side(False)
  eyes_l = "left_eye_region"
  eyes_r = "right_eye_region"
  lower = ["lower_face_region"]

  # --- region masks
  orbit_l = M.region(["left_orbital_region", "left_eye"], grow=1, soften=4)
  orbit_r = M.region(["right_orbital_region", "right_eye"], grow=1, soften=4)
  brow_l = M.region(["left_brow_region"], grow=3, soften=6)
  brow_r = M.region(["right_brow_region"], grow=3, soften=6)
  brows = M.region(["left_brow_region", "middle_brow_region",
                    "right_brow_region", "forehead_region"], grow=2, soften=6)
  inner_brow = brows * (1.0 - M.outer(0.012, 0.040))
  cheek_up = M.region(["left_infraorbital_region", "right_infraorbital_region",
                       "left_zygomatic_region", "right_zygomatic_region",
                       "left_cheek_region", "right_cheek_region"],
                      grow=2, soften=6)
  mouth = M.region(["upper_lip_region", "lower_lip_region", "upper_lip",
                    "lower_lip", "mouth_sock", "chin_region",
                    "left_cheek_region", "right_cheek_region"],
                   grow=3, soften=8)
  lips = M.region(["upper_lip", "lower_lip", "upper_lip_region",
                   "lower_lip_region", "mouth_sock"], grow=1, soften=4)
  nose = M.region(["nose_region"], grow=2, soften=6)
  cheeks = M.region(["left_cheek_region", "right_cheek_region",
                     "left_parotid_region", "right_parotid_region",
                     "upper_lip_region", "lower_lip_region", "upper_lip",
                     "lower_lip", "mouth_sock"], grow=3, soften=8)
  inner = M.group("tongue", "upper_teeth_and_gums", "lower_teeth_and_gums",
                  "mouth_sock")
  # just the lips (skin, no mouth interior), for the roll/close/lower shapes
  lip_skin = M.region(["upper_lip", "lower_lip"], grow=1, soften=5) * (
      1.0 - inner)

  # --- jaw field: how much each vertex follows the chin when the mouth
  # opens wide ('stretch_face'), from 0 (skull, upper lip, upper teeth) to 1
  # (chin, lower teeth). Splits lips upper/lower without any seam.
  open_d = field(coef["stretch_face"])
  low_teeth = M.group("lower_teeth_and_gums") > 0.5
  drop = -open_d[:, 1]
  J = np.clip(drop / max(np.median(drop[low_teeth]), 1e-6), 0.0, 1.0)
  J[M.group("upper_teeth_and_gums") > 0.5] = 0.0
  J[low_teeth] = 1.0
  J[M.group("tongue") > 0.5] = np.clip(J[M.group("tongue") > 0.5], 0.0, 1.0)
  J = M.smooth(J, 2)
  J[low_teeth] = 1.0
  J[M.group("upper_teeth_and_gums") > 0.5] = 0.0
  upper = (1.0 - J)
  jaw_zone = M.region(["chin_region", "lower_lip_region", "lower_lip",
                       "lower_teeth_and_gums", "tongue", "mouth_sock",
                       "left_cheek_region", "right_cheek_region",
                       "left_parotid_region", "right_parotid_region"],
                      grow=6, soften=10)
  Jz = J * jaw_zone

  # chin pivot (approx. jaw hinge below the ears) for the jaw rotations
  ear = M.group("ears") > 0.5
  hinge = np.array([0.0, M.v[ear, 1].mean() - 0.02, M.v[ear, 2].mean()])

  def jaw_rot(axis, deg):
    p = M.v - hinge
    return Jz[:, None] * (p @ _rot(axis, deg).T - p)

  # --- eyes: rigid eyeball rotations about GNM's eye joints
  ej = {"Left": gnm.joint_template[gnm.joint_names.index("left_eye")],
        "Right": gnm.joint_template[gnm.joint_names.index("right_eye")]}
  eyeball = {"Left": M.group("left_eye") > 0.5,
             "Right": M.group("right_eye") > 0.5}

  def look(side, direction):
    out_sign = 1.0 if side == "Left" else -1.0   # 'out' = towards the ear
    axis, deg = {
        "Up": ([-1.0, 0.0, 0.0], LOOK_DEG["Up"]),
        "Down": ([1.0, 0.0, 0.0], LOOK_DEG["Down"]),
        "Out": ([0.0, out_sign, 0.0], LOOK_DEG["Out"]),
        "In": ([0.0, -out_sign, 0.0], LOOK_DEG["In"]),
    }[direction]
    d = np.zeros_like(M.v)
    sel = eyeball[side]
    p = M.v[sel] - ej[side]
    d[sel] = p @ _rot(axis, deg).T - p
    return d

  eye_reg = {"Left": eyes_l, "Right": eyes_r}
  orbit = {"Left": orbit_l, "Right": orbit_r}
  brow = {"Left": brow_l, "Right": brow_r}
  half = {"Left": L, "Right": R}
  wink = {"Left": "wink_left", "Right": "wink_right"}

  def blink(s):
    return field(coef[wink[s]], [eye_reg[s]]) * orbit[s][:, None]

  def wide(s):
    return field(coef["surprise"], [eye_reg[s]]) * orbit[s][:, None] * 1.6

  def mouth_side(name, s, mask, gain=1.0):
    return field(coef[name]) * (mask * half[s])[:, None] * gain

  out = {}
  for s in ("Left", "Right"):
    out["eyeBlink" + s] = blink(s)
    out["eyeSquint" + s] = (field(coef["squint"], [eye_reg[s]])
                            * orbit[s][:, None] * 0.9)
    out["eyeWide" + s] = wide(s)
    out["eyeLookUp" + s] = look(s, "Up") + 0.35 * wide(s)
    out["eyeLookDown" + s] = look(s, "Down") + 0.25 * blink(s)
    out["eyeLookIn" + s] = look(s, "In")
    out["eyeLookOut" + s] = look(s, "Out")
    out["browOuterUp" + s] = (field(coef["surprise"], [eye_reg[s]])
                              * (brow[s] * M.outer(0.02, 0.045))[:, None]
                              * 3.0)
    out["browDown" + s] = (field(coef["compress_face"], [eye_reg[s]])
                           * (brow[s] * half[s])[:, None])
    out["cheekSquint" + s] = mouth_side("happy", s, cheek_up)
    out["noseSneer" + s] = mouth_side("snarl", s, nose, 1.3)
    out["mouthSmile" + s] = mouth_side("smile_wide", s, mouth, 1.1)
    out["mouthFrown" + s] = mouth_side("corners_down", s, mouth, 1.3)
    out["mouthPress" + s] = mouth_side("compress_face", s, lips)
    out["mouthStretch" + s] = mouth_side("platysma", s, mouth * upper
                                         + lips * J)
    out["mouthDimple" + s] = mouth_side("smile_wide", s,
                                        lips * M.outer(0.012, 0.03), 0.6)
    out["mouthUpperUp" + s] = mouth_side("snarl", s, lips * upper, 1.5)
    out["mouthLowerDown" + s] = (open_d * (lip_skin * J * half[s])[:, None]
                                 * 0.8)
  out["browInnerUp"] = (field(coef["surprise"], [eyes_l, eyes_r])
                        * inner_brow[:, None] * 2.6)
  out["cheekPuff"] = field(coef["blow"]) * cheeks[:, None]
  out["mouthPucker"] = field(coef["pucker"]) * mouth[:, None]
  out["mouthFunnel"] = field(coef["funneler"]) * mouth[:, None]
  roll = field(coef["lips_roll_in"])
  out["mouthRollLower"] = roll * (lip_skin * J)[:, None] * 0.6
  out["mouthRollUpper"] = roll * (lip_skin * upper)[:, None]
  out["mouthShrugLower"] = (field(coef["corners_down"])
                            * (mouth * J)[:, None])
  out["mouthShrugUpper"] = (field(coef["compress_face"])
                            * (lips * upper)[:, None])
  # mouthLeft = towards the subject's left (+X), like Anny's unit.
  ml, mr = field(coef["mouth_left"]), field(coef["mouth_right"])
  lip_v = M.group("upper_lip", "lower_lip") > 0.5
  if ml[lip_v, 0].mean() < mr[lip_v, 0].mean():
    ml, mr = mr, ml
  out["mouthLeft"] = ml * mouth[:, None] * 0.6
  out["mouthRight"] = mr * mouth[:, None] * 0.6
  jaw_open = open_d * Jz[:, None] * 1.3
  out["jawOpen"] = jaw_open
  # mouthClose (used with jawOpen): the lips meet in between - the lower
  # lip comes up 60% of the gap, the upper lip down 40% (less stretch).
  low_lip = M.group("lower_lip") > 0.5
  gap = jaw_open[low_lip].mean(axis=0)
  out["mouthClose"] = (-0.6 * jaw_open * (lip_skin * J)[:, None]
                       + 0.4 * gap[None, :] * (lip_skin * upper)[:, None])
  out["jawLeft"] = jaw_rot([0.0, 1.0, 0.0], 3.0) + Jz[:, None] * [0.003, 0, 0]
  out["jawRight"] = (jaw_rot([0.0, -1.0, 0.0], 3.0)
                     + Jz[:, None] * [-0.003, 0, 0])
  out["jawForward"] = Jz[:, None] * np.array([0.0, 0.0, 0.007])
  out["tongueOut"] = field(coef["tongue_center"], ["tongue"])
  missing = [l for l in labels if l not in out]
  if missing:
    raise RuntimeError("no GNM recipe for %s" % ", ".join(missing))
  lam = float(os.environ.get("ARKIT_LAM", PROJECT_LAMBDA))
  if lam > 0:
    P = _Projector(B, lam)
    for l in out:
      if not l.startswith("eyeLook") and l != "tongueOut":
        out[l] = P(out[l])
  return np.stack([out[l] for l in labels]).astype(np.float32)


class _Projector(object):
  """Ridge projection of a delta field onto the span of GNM's expression
  basis: whatever the masks did, the result is a combination of GNM modes
  (learned from real scans), so lips and lids stay attached."""

  def __init__(self, B, lam_rel):
    self.Bf = B.reshape(B.shape[0], -1).astype(np.float64)
    A = self.Bf @ self.Bf.T
    A[np.diag_indices_from(A)] += lam_rel * np.trace(A) / len(A)
    self.Ainv = np.linalg.inv(A)

  def __call__(self, T):
    c = self.Ainv @ (self.Bf @ np.asarray(T, np.float64).reshape(-1))
    return (c @ self.Bf).reshape(T.shape)


class ArkitDeltas(object):
  """The compiled per-ARKit GNM deltas; ``delta(weights)`` sums them."""

  def __init__(self, labels, deltas):
    self.labels = list(labels)
    self.index = {l: i for i, l in enumerate(self.labels)}
    self.deltas = deltas

  def delta(self, weights):
    """(V, 3) GNM-frame displacement for {label: weight}, or None."""
    w = [(self.index[k], float(v)) for k, v in (weights or {}).items()
         if v and k in self.index]
    if not w:
      return None
    idx = np.array([i for i, _ in w])
    val = np.array([v for _, v in w])
    return np.einsum("n,nvk->vk", val, self.deltas[idx].astype(np.float64))


def ensure_arkit(gnm_loader, gnm_root, labels, log=None):
  """ArkitDeltas for this install, compiled once (needs h5py) and cached
  next to the GNM data. ``gnm_loader`` returns a GNMModel."""
  path = os.path.join(gnm_root, ARKIT_COMPILED_NAME)
  if os.path.isfile(path):
    try:
      with np.load(path, allow_pickle=False) as z:
        if (int(z["version"]) == ARKIT_COMPILED_VERSION
            and [str(x) for x in z["labels"]] == list(labels)):
          return ArkitDeltas(labels, z["deltas"])
    except Exception:
      pass
  if log:
    log("Building the ARKit shapes for the GNM head (one time)...")
  deltas = compile_arkit(gnm_loader(), os.path.join(gnm_root, DECODER_REL),
                         labels)
  tmp = path + ".tmp.npz"
  np.savez_compressed(tmp, version=ARKIT_COMPILED_VERSION,
                      labels=np.asarray(list(labels)), deltas=deltas)
  os.replace(tmp, path)
  return ArkitDeltas(labels, deltas)


def decoder_available(gnm_root):
  try:
    import h5py  # noqa: F401
  except Exception:
    return False
  return os.path.isfile(os.path.join(gnm_root, DECODER_REL))
