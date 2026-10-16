"""Check external/_head.GNMModel against Google's official gnm_numpy.

DEV-ONLY. Run with a Python that can import the official ``gnm`` package
(e.g. gnm-maya's runtime, which vendors google/GNM with its deps):

  <gnm-maya>/runtime/python.exe anny-maya/tests/validate_gnm.py \
      --gnm-repo <path to google/GNM checkout at the pinned commit>

Compares the bind-pose vertices (template + identity + expression) and the
zero-pose official output for random coefficients.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "external"))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gnm-repo", required=True)
  ap.add_argument("--cases", type=int, default=20)
  a = ap.parse_args()
  sys.path.insert(0, a.gnm_repo)
  from gnm.shape import gnm_numpy as G
  import _head

  official = G.GNM.from_local(version=G.GNMMajorVersion.V3,
                              variant=G.GNMVariant.HEAD)
  npz = os.path.join(a.gnm_repo, _head.GNM_NPZ_REL)
  ours = _head.GNMModel(npz)
  rng = np.random.default_rng(7)
  worst_bind = worst_world = worst_joints = 0.0
  for k in range(a.cases):
    ident = rng.normal(size=ours.identity_dim) * (k % 3)
    expr = rng.normal(size=ours.expression_dim) * (k % 2)
    v, j = ours.bind(ident, expr)
    ref_bind = np.asarray(official.vertex_positions_bind_pose(
        ident.astype(np.float32), expr.astype(np.float32)), np.float64)
    ref_world = np.asarray(official(identity=ident.astype(np.float32),
                                    expression=expr.astype(np.float32)),
                           np.float64)
    ref_j = np.asarray(official.joint_positions_bind_pose(
        ident.astype(np.float32)), np.float64)
    worst_bind = max(worst_bind, np.abs(v - ref_bind).max())
    worst_world = max(worst_world, np.abs(v - ref_world).max())
    worst_joints = max(worst_joints, np.abs(j - ref_j).max())
  print("GNM bind vertices   max |err| = %.2e m" % worst_bind)
  print("GNM zero-pose mesh  max |err| = %.2e m" % worst_world)
  print("GNM joints          max |err| = %.2e m" % worst_joints)
  # The official model runs in float32; ours in float64.
  ok = max(worst_bind, worst_world, worst_joints) < 1e-5
  print("RESULT: %s" % ("PASS" if ok else "FAIL"))
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
