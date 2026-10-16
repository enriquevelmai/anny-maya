"""High-level Anny body controller used by the UI and user scripts.

Holds the full parameter state in plain Python (no numpy in mayapy), drives
the resident worker, and pushes results into one Maya mesh.

State (all JSON-friendly, persisted on the mesh transform):
  phenotypes      {gender, age, muscle, weight, height, proportions, cupsize,
                   firmness, african, asian, caucasian} -> float (0.5 neutral)
  local_changes   {MakeHuman target label -> -1..1}  (Body tab)
  facial_actions  {ARKit name -> 0..1}               (Face tab)
  rotations       {bone label -> [rx, ry, rz]} degrees about Maya world axes
                  at the neutral pose (Anny's local-ref parameterization)
  translation     [x, y, z] cm (root)
"""

from __future__ import annotations

import copy
import json
import logging
import random

from maya import cmds as mc

from anny_maya.core import worker as _worker
from anny_maya.scene import build as _build

logger = logging.getLogger(__name__)

STATE_ATTR = "annyState"

PHENOTYPES = ["gender", "age", "muscle", "weight", "height", "proportions",
              "cupsize", "firmness", "african", "asian", "caucasian"]
AGE_MIN = -1.0 / 3.0  # newborn anchor; 0 = baby, 1/3 child, 2/3 young, 1 old


def find_bodies(selected_only=False):
  """Anny body transforms in the scene (identified by the annyState attr)."""
  nodes = (mc.ls(selection=True, long=False) if selected_only
           else mc.ls(type="transform", long=False))
  return [n for n in (nodes or [])
          if mc.attributeQuery(STATE_ATTR, node=n, exists=True)
          and not mc.attributeQuery("annyRig", node=n, exists=True)]


def default_state():
  return {"phenotypes": {k: 0.5 for k in PHENOTYPES},
          "local_changes": {}, "facial_actions": {}, "rotations": {},
          "translation": [0.0, 0.0, 0.0]}


def _sym_map(pairs):
  m = {}
  for a, b in pairs:
    m[a] = b
    m[b] = a
  return m


class AnnyBody(object):
  """A live Anny body: mutate parameters, the mesh updates in place."""

  def __init__(self, name="anny_body", state=None):
    self._init_common()
    if state:
      self.set_state(state, update=False)
    verts = self.worker.eval(self.state)
    self.transform = _build.build_mesh(self.topology, verts, name=name)
    self._save_state()
    logger.info("Built Anny body '%s'", self.transform)

  @classmethod
  def adopt(cls, transform):
    """Bind a controller to an EXISTING body, restoring its saved state."""
    self = cls.__new__(cls)
    self._init_common()
    self.transform = transform
    restored = self._load_state()
    if restored:
      self._update()  # make sure the mesh matches the restored parameters
    logger.info("Adopted Anny body '%s' (%s)", transform,
                "state restored" if restored else "no saved state")
    return self

  def _init_common(self):
    self.worker = _worker.get_worker()
    self.topology = self.worker.topology
    meta = self.topology.meta
    self.state = default_state()
    self.bone_labels = list(meta["bone_labels"])
    self.local_labels = list(meta["local_labels"])
    self.face_labels = list(meta["face_labels"])
    self.bone_mirror = _sym_map(meta.get("bone_mirror", []))
    self.local_mirror = _sym_map(meta.get("local_mirror", []))
    self.face_mirror = _sym_map(meta.get("face_mirror", []))

  # --- state persistence ---------------------------------------------------

  def _save_state(self):
    if not mc.objExists(self.transform):
      return
    if not mc.attributeQuery(STATE_ATTR, node=self.transform, exists=True):
      mc.addAttr(self.transform, longName=STATE_ATTR, dataType="string")
    mc.setAttr("%s.%s" % (self.transform, STATE_ATTR),
               json.dumps(self.state), type="string")

  def _load_state(self):
    if not mc.attributeQuery(STATE_ATTR, node=self.transform, exists=True):
      return False
    raw = mc.getAttr("%s.%s" % (self.transform, STATE_ATTR))
    if not raw:
      return False
    try:
      self.set_state(json.loads(raw), update=False)
      return True
    except Exception:
      logger.exception("Ignoring corrupt %s on '%s'", STATE_ATTR,
                       self.transform)
      return False

  def get_state(self):
    return copy.deepcopy(self.state)

  def set_state(self, state, update=True):
    s = default_state()
    s["phenotypes"].update({k: float(v) for k, v in
                            (state.get("phenotypes") or {}).items()
                            if k in s["phenotypes"]})
    s["local_changes"] = {k: float(v) for k, v in
                          (state.get("local_changes") or {}).items()
                          if k in self.local_labels and float(v) != 0.0}
    s["facial_actions"] = {k: float(v) for k, v in
                           (state.get("facial_actions") or {}).items()
                           if k in self.face_labels and float(v) != 0.0}
    s["rotations"] = {k: [float(a) for a in v] for k, v in
                      (state.get("rotations") or {}).items()
                      if k in self.bone_labels and any(float(a) for a in v)}
    s["translation"] = [float(x) for x in
                        (state.get("translation") or [0.0, 0.0, 0.0])]
    self.state = s
    if update:
      self._update()

  # --- evaluation ----------------------------------------------------------

  def _update(self):
    verts = self.worker.eval(self.state)
    if not mc.objExists(self.transform):
      # The user deleted the mesh: self-heal by rebuilding it with the
      # current parameters instead of erroring on every button press.
      old = self.transform
      self.transform = _build.build_mesh(self.topology, verts,
                                         name=old.split("|")[-1])
      logger.info("Mesh '%s' was deleted - recreated as '%s'", old,
                  self.transform)
    else:
      _build.set_points(self.transform, verts)
    self._save_state()

  def refresh(self):
    """Re-evaluate and push the mesh (callers stage edits with update=False
    and repaint once - the UI throttles slider drags this way)."""
    self._update()

  # --- single-parameter edits ---------------------------------------------

  def set_phenotype(self, label, value, update=True):
    if label not in self.state["phenotypes"]:
      raise ValueError("unknown phenotype %r" % label)
    self.state["phenotypes"][label] = float(value)
    if update:
      self._update()

  @staticmethod
  def _put(d, key, value):
    if float(value) == 0.0:
      d.pop(key, None)
    else:
      d[key] = float(value)

  def set_local(self, label, value, symmetry=False, update=True):
    """Set one local change (-1..1). Returns the labels that changed."""
    changed = [label]
    self._put(self.state["local_changes"], label, value)
    if symmetry and label in self.local_mirror:
      m = self.local_mirror[label]
      self._put(self.state["local_changes"], m, value)
      changed.append(m)
    if update:
      self._update()
    return changed

  def set_face(self, label, value, symmetry=False, update=True):
    """Set one ARKit facial action (0..1). Returns the labels changed."""
    changed = [label]
    self._put(self.state["facial_actions"], label, value)
    if symmetry and label in self.face_mirror:
      m = self.face_mirror[label]
      self._put(self.state["facial_actions"], m, value)
      changed.append(m)
    if update:
      self._update()
    return changed

  @staticmethod
  def mirror_rotation(rot):
    """Reflect a world-axis XYZ rotation across Maya's YZ plane."""
    return [rot[0], -rot[1], -rot[2]]

  def rotation(self, bone):
    return list(self.state["rotations"].get(bone, [0.0, 0.0, 0.0]))

  def set_rotation(self, bone, axis, value, symmetry=False, update=True):
    """Set one bone rotation axis (degrees). Returns changed bones."""
    rot = self.rotation(bone)
    rot[axis] = float(value)
    self._store_rotation(bone, rot)
    changed = [bone]
    if symmetry and bone in self.bone_mirror:
      m = self.bone_mirror[bone]
      self._store_rotation(m, self.mirror_rotation(rot))
      changed.append(m)
    if update:
      self._update()
    return changed

  def _store_rotation(self, bone, rot):
    if any(abs(a) > 0 for a in rot):
      self.state["rotations"][bone] = [float(a) for a in rot]
    else:
      self.state["rotations"].pop(bone, None)

  def set_translation(self, axis, value, update=True):
    self.state["translation"][axis] = float(value)
    if update:
      self._update()

  # --- bulk ops ------------------------------------------------------------

  def randomize_phenotypes(self, seed=None, children=False, extra=False,
                           update=True):
    """Plausible random body: gender mostly near male/female, adult ages
    unless ``children``; ``extra`` also rolls cup size/firmness/ancestry."""
    rng = random.Random(seed)
    clip = lambda x, lo=0.0, hi=1.0: max(lo, min(hi, x))
    p = self.state["phenotypes"]
    g = 0.0 if rng.random() < 0.5 else 1.0
    p["gender"] = clip(abs(g - abs(rng.gauss(0.0, 0.12))))
    p["age"] = (rng.uniform(AGE_MIN, 1.0) if children
                else rng.uniform(0.55, 0.95))
    p["muscle"] = clip(rng.gauss(0.5, 0.18))
    p["weight"] = clip(rng.gauss(0.5, 0.18))
    p["height"] = clip(rng.gauss(0.5, 0.2))
    p["proportions"] = clip(rng.gauss(0.5, 0.2))
    if extra:
      p["cupsize"] = clip(rng.gauss(0.5, 0.2))
      p["firmness"] = clip(rng.gauss(0.5, 0.2))
      w = [rng.random() ** 2 for _ in range(3)]
      s = sum(w) or 1.0
      for k, x in zip(("african", "asian", "caucasian"), w):
        p[k] = x / s
    if update:
      self._update()

  def randomize_local(self, scale=0.3, seed=None, symmetric=True,
                      labels=None, update=True):
    """Random local changes (all, or just ``labels``) within +-scale."""
    rng = random.Random(seed)
    labels = list(labels) if labels is not None else self.local_labels
    done = set()
    for lab in labels:
      if lab in done:
        continue
      v = rng.uniform(-1.0, 1.0) * scale
      self._put(self.state["local_changes"], lab, v)
      done.add(lab)
      if symmetric and lab in self.local_mirror:
        m = self.local_mirror[lab]
        self._put(self.state["local_changes"], m, v)
        done.add(m)
    if update:
      self._update()

  def randomize_face(self, scale=0.6, seed=None, symmetric=True, count=6,
                     update=True):
    """A random expression: ``count`` face units at random strengths."""
    rng = random.Random(seed)
    self.state["facial_actions"] = {}
    for lab in rng.sample(self.face_labels, min(count, len(self.face_labels))):
      if lab.startswith("eyeLook") or lab == "tongueOut":
        continue  # gaze/tongue make random faces look broken
      v = rng.uniform(0.2, 1.0) * scale
      self._put(self.state["facial_actions"], lab, v)
      if symmetric and lab in self.face_mirror:
        self._put(self.state["facial_actions"], self.face_mirror[lab], v)
    if update:
      self._update()

  def apply_pose(self, rotations, keep_others=False, update=True):
    if not keep_others:
      self.state["rotations"] = {}
    for bone, rot in rotations.items():
      if bone in self.bone_labels:
        self._store_rotation(bone, [float(a) for a in rot])
    if update:
      self._update()

  def pose_preset(self, name, update=True):
    """'neutral', 'tpose' or 'arms_down' (aimed by the worker for THIS
    body's proportions). Keeps non-arm bones as they are."""
    rots = self.worker.pose_preset(name, self.state)
    arm = ("clavicle", "shoulder", "upperarm", "lowerarm", "wrist",
           "finger", "metacarpal")
    kept = {b: r for b, r in self.state["rotations"].items()
            if not b.startswith(arm)}
    kept.update(rots)
    self.state["rotations"] = {}
    self.apply_pose(kept, update=update)

  def mirror_pose(self, from_side="L", update=True):
    """Copy every .L (or .R) bone rotation onto its mirror."""
    suffix = "." + from_side
    for bone, rot in list(self.state["rotations"].items()):
      if bone.endswith(suffix) and bone in self.bone_mirror:
        self._store_rotation(self.bone_mirror[bone], self.mirror_rotation(rot))
    for bone in self.bone_labels:  # zeroed source bones zero their mirror too
      if bone.endswith(suffix) and bone not in self.state["rotations"]:
        m = self.bone_mirror.get(bone)
        if m:
          self.state["rotations"].pop(m, None)
    if update:
      self._update()

  def reset_phenotypes(self, update=True):
    self.state["phenotypes"] = default_state()["phenotypes"]
    if update:
      self._update()

  def reset_local(self, labels=None, update=True):
    if labels is None:
      self.state["local_changes"] = {}
    else:
      for lab in labels:
        self.state["local_changes"].pop(lab, None)
    if update:
      self._update()

  def reset_face(self, update=True):
    self.state["facial_actions"] = {}
    if update:
      self._update()

  def reset_pose(self, update=True):
    self.state["rotations"] = {}
    self.state["translation"] = [0.0, 0.0, 0.0]
    if update:
      self._update()

  def reset_all(self):
    self.state = default_state()
    self._update()
    logger.info("Reset all parameters on '%s'", self.transform)


# --- convenience one-shots (no persistent controller) ----------------------

def create_body(seed=None, randomize=True, name="anny_body"):
  """Build a body (random unless ``randomize=False``) and return its
  AnnyBody controller. ``seed=None`` gives a different body each call."""
  b = AnnyBody(name=name)
  if randomize:
    b.randomize_phenotypes(seed=seed)
  return b
