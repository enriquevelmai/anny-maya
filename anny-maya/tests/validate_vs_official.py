"""Regression check: numpy port (external/_anny_core.py) vs official Anny.

DEV-ONLY. Never shipped to users and never imported by the Maya module.

  1. generate  — needs a dev venv with the official ``anny`` package (pinned
                 release, see external/ANNY_DATA.json) + torch CPU. Builds the
                 official model, draws random phenotypes / local changes /
                 facial actions / poses (seeded), and writes reference
                 fixtures (``_fixtures/*.npz``, gitignored).
  2. check     — needs only numpy. Builds the numpy port from the same data
                 folder and asserts every fixture matches vertex for vertex.

    python tests/validate_vs_official.py all --data <anny>/src/anny/data
    python tests/validate_vs_official.py check --data <dir>   # numpy only

The data folder defaults to the module's downloaded copy
(external/anny_data/src/anny/data) when present.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MODULE, "external"))

FIXTURES = os.path.join(MODULE, "_fixtures")
TOL = 1e-9  # metres; float64 everywhere, so the port should be ~1e-13 off

# (name, phenotype set, n cases, pose style)
SUITES = [
    ("neutral", "default", 1, "none"),
    ("phenotypes", "default", 12, "none"),
    ("phenotypes_all", "all", 8, "none"),
    ("extremes", "default", 8, "none"),
    ("local_changes", "default", 10, "none"),
    ("facial_actions", "default", 10, "none"),
    ("pose_small", "default", 10, "small"),
    ("pose_random", "default", 6, "random"),
    ("everything", "all", 10, "random"),
]


def _default_data_dir():
  p = os.path.join(MODULE, "external", "anny_data", "src", "anny", "data")
  return p if os.path.isdir(p) else None


def _random_rotations(rng, n, scale):
  """Rotation matrices from random rotation vectors (|angle| <= scale*pi)."""
  out = np.zeros((n, 3, 3))
  for i in range(n):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(-1, 1) * scale * np.pi
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    out[i] = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
  return out


def _make_cases(suite, nbones, local_labels, face_labels, phen_labels, seed):
  name, _phen, n, pose = suite
  rng = np.random.default_rng(seed)
  cases = []
  for k in range(n):
    c = {"phenotypes": {}, "local_changes": {}, "facial_actions": {},
         "pose": None}
    if name == "extremes":
      c["phenotypes"] = {l: float(rng.choice([-0.5, 0.0, 1.0, 1.5]))
                         for l in phen_labels}
    elif name != "neutral":
      c["phenotypes"] = {l: float(rng.uniform(0, 1)) for l in phen_labels}
      if "age" in c["phenotypes"] and rng.uniform() < 0.3:
        c["phenotypes"]["age"] = float(rng.uniform(-1 / 3, 0.3))
    if name in ("local_changes", "everything"):
      for l in rng.choice(local_labels, size=12, replace=False):
        c["local_changes"][str(l)] = float(rng.uniform(-1, 1))
    if name in ("facial_actions", "everything"):
      for l in rng.choice(face_labels, size=10, replace=False):
        c["facial_actions"][str(l)] = float(rng.uniform(0, 1))
    if pose != "none":
      scale = 0.15 if pose == "small" else 1.0
      rot = _random_rotations(rng, nbones, scale)
      d = np.tile(np.eye(4), (nbones, 1, 1))
      d[:, :3, :3] = rot
      d[0, :3, 3] = rng.normal(size=3) * 0.2
      c["pose"] = d.tolist()
    cases.append(c)
  return cases


def generate(data_dir, seed=1234):
  import torch
  import anny
  import anny.paths
  from anny.models.facial_actions import FACIAL_ACTION_LABELS

  anny.paths.set_anny_root_dir(os.path.dirname(os.path.abspath(data_dir)))
  os.makedirs(FIXTURES, exist_ok=True)
  models = {}
  for si, suite in enumerate(SUITES):
    phen = suite[1]
    if phen not in models:
      t = time.time()
      models[phen] = anny.Anny(rig="anny", topology="anny-quads",
                               local_changes="default", facial_actions="all",
                               phenotypes=phen, skinning_method="lbs")
      print("official model (%s) built in %.1fs" % (phen, time.time() - t))
    m = models[phen]
    cases = _make_cases(suite, m.bone_count, m.local_change_labels,
                        list(FACIAL_ACTION_LABELS), m.phenotype_labels,
                        seed + si)
    for k, c in enumerate(cases):
      kw = {}
      if c["phenotypes"]:
        kw["phenotype_kwargs"] = {a: torch.tensor([v], dtype=torch.float64)
                                  for a, v in c["phenotypes"].items()}
      if c["local_changes"]:
        kw["local_changes_kwargs"] = {a: torch.tensor([v], dtype=torch.float64)
                                      for a, v in c["local_changes"].items()}
      if c["facial_actions"]:
        kw["facial_actions"] = {a: torch.tensor([v], dtype=torch.float64)
                                for a, v in c["facial_actions"].items()}
      if c["pose"] is not None:
        kw["pose_parameters"] = torch.tensor([c["pose"]], dtype=torch.float64)
      with torch.no_grad():
        out = m(**kw)
        coeffs = m._get_phenotype_blendshape_coefficients(
            *m.get_tensor_inputs(None, kw.get("phenotype_kwargs"),
                                 kw.get("local_changes_kwargs"),
                                 kw.get("facial_actions"))[1:])
      path = os.path.join(FIXTURES, "%s_%02d.npz" % (suite[0], k))
      np.savez_compressed(
          path, case=np.asarray(json.dumps(c)), phenotypes=np.asarray(phen),
          coeffs=coeffs[0].numpy(),
          rest_vertices=out["rest_vertices"][0].numpy(),
          rest_bone_poses=out["rest_bone_poses"][0].numpy(),
          bone_poses=out["bone_poses"][0].numpy(),
          vertices=out["vertices"][0].numpy())
  m = models["default"]
  dense_w = np.zeros((m.template_vertices.shape[0], m.bone_count))
  np.add.at(dense_w, (np.arange(dense_w.shape[0])[:, None],
                      m.vertex_bone_indices.numpy()),
            m.vertex_bone_weights.numpy())
  np.savez_compressed(
      os.path.join(FIXTURES, "static.npz"),
      template=m.template_vertices.numpy(), faces=m.faces.numpy(),
      uvs=m.texture_coordinates.numpy(),
      face_uv=m.face_texture_coordinate_indices.numpy(),
      weights=dense_w, bone_parents=np.asarray(m.bone_parents),
      bone_labels=np.asarray(json.dumps(list(m.bone_labels))),
      bs_labels=np.asarray(json.dumps(list(m.blendshape_labels))),
      local_labels=np.asarray(json.dumps(list(m.local_change_labels))),
      base_vertex_indices=m.base_mesh_vertex_indices.numpy())
  print("fixtures written to %s" % FIXTURES)


def check(data_dir):
  import _anny_core as core

  t = time.time()
  arrays = core.build_model(data_dir)
  print("numpy model built in %.1fs" % (time.time() - t))
  st = np.load(os.path.join(FIXTURES, "static.npz"))
  ok = True

  def report(label, err, tol=TOL):
    nonlocal ok
    good = err <= tol
    ok &= good
    print("  %-34s max|err| = %.3e %s" % (label, err, "ok" if good else "FAIL"))
    return err

  print("static data:")
  report("template vertices", np.abs(arrays["template"] - st["template"]).max())
  same = (np.array_equal(arrays["faces"], st["faces"])
          and np.array_equal(arrays["face_uv"], st["face_uv"]))
  report("faces + uv indices (exact)", 0.0 if same else 1.0)
  report("uvs", np.abs(arrays["uvs"] - st["uvs"]).max())
  report("skin weights (dense)", np.abs(arrays["weights"] - st["weights"]).max())
  same = (list(arrays["bone_labels"]) == json.loads(str(st["bone_labels"]))
          and list(arrays["bone_parents"]) == list(st["bone_parents"])
          and list(arrays["bs_labels"]) == json.loads(str(st["bs_labels"]))
          and list(arrays["local_labels"]) == json.loads(str(st["local_labels"]))
          and np.array_equal(arrays["base_vertex_indices"],
                             st["base_vertex_indices"]))
  report("bone/blendshape labels + order", 0.0 if same else 1.0)

  worst = {"rest_vertices": 0.0, "rest_bone_poses": 0.0, "bone_poses": 0.0,
           "vertices": 0.0, "coeffs": 0.0}
  model = core.AnnyModel(arrays)
  files = sorted(glob.glob(os.path.join(FIXTURES, "*_[0-9][0-9].npz")))
  times = []
  for path in files:
    f = np.load(path)
    c = json.loads(str(f["case"]))
    coeffs = model.coefficients(c["phenotypes"], c["local_changes"],
                                c["facial_actions"])
    deltas = None if c["pose"] is None else np.asarray(c["pose"])
    t = time.time()
    out = model.evaluate(coeffs, deltas)
    times.append(time.time() - t)
    errs = {"coeffs": np.abs(coeffs - f["coeffs"]).max()}
    for k in ("rest_vertices", "rest_bone_poses", "bone_poses", "vertices"):
      errs[k] = np.abs(out[k] - f[k]).max()
    for k, e in errs.items():
      worst[k] = max(worst[k], e)
    if max(errs.values()) > TOL:
      ok = False
      print("  FAIL %s: %s" % (os.path.basename(path),
                              {k: "%.2e" % v for k, v in errs.items()}))
  print("%d evaluated fixtures (random phenotypes, local changes, facial "
        "actions, poses):" % len(files))
  for k, e in worst.items():
    report("worst " + k, e)
  print("numpy eval: %.1f ms median" % (1000 * float(np.median(times))))
  print("RESULT: %s" % ("PASS" if ok else "FAIL"))
  return ok


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("cmd", choices=["generate", "check", "all"])
  ap.add_argument("--data", default=_default_data_dir(),
                  help="extracted naver/anny src/anny/data folder")
  ap.add_argument("--seed", type=int, default=1234)
  a = ap.parse_args()
  if not a.data:
    ap.error("--data is required (no downloaded copy in the module)")
  if a.cmd in ("generate", "all"):
    generate(a.data, a.seed)
  if a.cmd in ("check", "all"):
    sys.exit(0 if check(a.data) else 1)


if __name__ == "__main__":
  main()
