"""Headless validation under mayapy of the whole Maya pipeline.

    mayapy -c "import sys; sys.path.insert(0, r'<module>/scripts'); \
               from anny_maya import smoke_test; smoke_test.run_headless()"

Covers: quad mesh + UVs + materials, every parameter family driving the
mesh, symmetry, state persistence/adoption, pose presets, the baked rig
(joints + skinCluster + 52 ARKit blendShapes) reproducing the worker's
evaluation exactly, FBX export, and building the real panel.
"""

from __future__ import annotations

import os
import sys
import tempfile


def _pts(name, world=True):
  from anny_maya.scene.build import get_points
  return get_points(name, world=world)


def _max_diff(a, b):
  return max(abs(x - y) for x, y in zip(a, b))


def _moved(a, b, tol=1e-4):
  """Vertex indices that moved more than ``tol`` between two flat lists."""
  return [i // 3 for i in range(0, len(a), 3)
          if max(abs(a[i] - b[i]), abs(a[i + 1] - b[i + 1]),
                 abs(a[i + 2] - b[i + 2])) > tol]


def run_headless():
  """Entry point for CI: create the GUI QApplication BEFORE Maya standalone
  initializes (the other order aborts), then run the full test."""
  try:
    from PySide6 import QtWidgets
  except ImportError:
    from PySide2 import QtWidgets
  QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
  try:
    sys.stdout.reconfigure(line_buffering=True)
  except Exception:
    pass
  return run()


def run():
  import maya.standalone
  maya.standalone.initialize(name="python")
  try:
    _run()
  finally:
    try:
      from anny_maya.core import worker
      worker.shutdown_worker()
    except Exception:
      pass
    maya.standalone.uninitialize()


def _run():
  import time
  from maya import cmds as mc
  from anny_maya.services import bootstrap
  from anny_maya.core.body import AnnyBody, find_bodies
  from anny_maya.core import worker

  assert bootstrap.all_available() or bootstrap.ensure_all_with_dialog(), \
      "first-run setup failed"
  mc.file(new=True, force=True)

  t = time.time()
  body = AnnyBody(name="anny_smoke")
  name = body.transform
  topo = body.topology
  print("[ok] body built in %.2fs (worker start + first eval + mesh)"
        % (time.time() - t))
  assert mc.polyEvaluate(name, face=True) == topo.num_faces == 13710
  assert mc.polyEvaluate(name, vertex=True) == topo.num_vertices == 13718
  import maya.api.OpenMaya as om2
  sel = om2.MSelectionList()
  sel.add(name)
  fn = om2.MFnMesh(sel.getDagPath(0))
  assert fn.polygonVertexCount(0) == 4 and fn.numUVs() > 0
  print("[ok] quad mesh: %d verts, %d quads, %d UVs"
        % (fn.numVertices, fn.numPolygons, fn.numUVs()))
  for sg in ("anny_skin_matSG", "anny_eyes_matSG", "anny_tongue_matSG"):
    assert mc.objExists(sg), "missing shading group %s" % sg
    assert mc.nodeType(mc.listConnections(sg + ".surfaceShader")[0]) in (
        "standardSurface", "lambert")
  print("[ok] standardSurface materials: skin / eyes / tongue")

  base = _pts(name)
  ys = base[1::3]
  height = max(ys) - min(ys)
  assert abs(min(ys)) < 0.05, "feet not on the ground (min y=%.3f)" % min(ys)
  assert 140.0 < height < 200.0, "implausible height %.1f cm" % height
  print("[ok] neutral body: %.1f cm tall, feet at y=%.3f" % (height, min(ys)))

  # --- every parameter family drives the mesh -------------------------------
  t = time.time()
  n = 20
  for i in range(n):
    body.set_phenotype("weight", i / float(n))
  dt = (time.time() - t) / n
  body.set_phenotype("weight", 0.5)
  print("[ok] interactive eval + mesh push: %.1f ms per edit" % (dt * 1000))
  body.set_phenotype("age", 0.0)
  kid = _pts(name)
  kys = kid[1::3]
  assert max(kys) - min(kys) < height * 0.75, "age did not shrink the body"
  body.set_phenotype("age", 0.5)
  print("[ok] phenotype age: baby is %.1f cm" % (max(kys) - min(kys)))

  before = _pts(name)
  changed = body.set_local("l-upperarm-fat-incr", 1.0, symmetry=True)
  assert set(changed) == {"l-upperarm-fat-incr", "r-upperarm-fat-incr"}
  moved = _moved(before, _pts(name))
  xs = [before[3 * v] for v in moved]
  assert moved and min(xs) < 0 < max(xs), "symmetric local change one-sided"
  body.reset_local()
  print("[ok] local change + symmetry: %d verts moved on both sides"
        % len(moved))

  before = _pts(name)
  body.set_face("jawOpen", 1.0)
  moved = _moved(before, _pts(name))
  top = 0.8 * max(before[1::3])  # well below the chin
  assert moved and all(before[3 * v + 1] > top for v in moved), \
      "jawOpen moved vertices outside the head"
  body.reset_face()
  print("[ok] facial action jawOpen: %d head verts moved" % len(moved))

  before = _pts(name)
  body.set_rotation("upperarm01.L", 2, 40.0, symmetry=True)
  assert "upperarm01.R" in body.state["rotations"]
  assert body.state["rotations"]["upperarm01.R"] == [0.0, -0.0, -40.0]
  moved = _moved(before, _pts(name))
  assert len(moved) > 500, "pose did not move the arms"
  body.reset_pose()
  print("[ok] pose + mirrored rotation: %d verts moved" % len(moved))

  body.pose_preset("tpose")
  meta = body.worker.bake(body.get_state())
  lab = meta["bone_labels"]

  def wpos(label):
    m = meta["posed_world"][lab.index(label)]
    return m[12], m[13], m[14]
  s, w = wpos("upperarm01.L"), wpos("wrist.L")
  assert abs(w[1] - s[1]) < 3.0 and abs(w[2] - s[2]) < 3.0 and w[0] > s[0], \
      "T-pose arm not horizontal: shoulder %s wrist %s" % (s, w)
  body.reset_pose()
  print("[ok] T-pose preset: left wrist level with the shoulder "
        "(dy=%.2f cm)" % (w[1] - s[1]))

  # --- persistence ----------------------------------------------------------
  body.set_phenotype("muscle", 0.8)
  body.set_local("nose-scale-vert-incr", 0.5)
  twin = AnnyBody.adopt(name)
  assert twin.state == body.state, "state not restored on adopt"
  assert name in find_bodies()
  print("[ok] state persists on the mesh and is restored on adopt")

  # --- baked rig reproduces the worker exactly -----------------------------------
  from anny_maya.scene import rig
  body.randomize_phenotypes(seed=5)
  body.set_local("l-lowerarm-fat-incr", 0.7, symmetry=True)
  body.set_face("mouthSmileLeft", 0.8)
  body.set_face("eyeBlinkRight", 0.6)
  body.apply_pose({"upperarm01.L": [10, 20, 50], "lowerarm01.L": [0, 60, 0],
                   "spine03": [15, 0, 5], "head": [-20, 30, 0],
                   "upperleg01.R": [-40, 0, 10], "lowerleg01.R": [70, 0, 0],
                   "finger2-1.L": [0, 0, -45], "root": [0, 30, 0]})
  body.set_translation(0, 12.0)
  t = time.time()
  rigged = rig.bake_rig(body)
  print("[ok] baked rig in %.1fs" % (time.time() - t))
  info = rig.rig_info(rigged)
  joints = rig.rig_joints(rigged)
  assert len(joints) == 104 and mc.objExists(info["skinCluster"])
  bs = info["blendShape"]
  aliases = mc.aliasAttr(bs, query=True)[::2]
  for want in ("eyeBlinkLeft", "jawOpen", "mouthSmileRight", "tongueOut"):
    assert want in aliases, "missing ARKit target %s" % want
  assert len(aliases) == 52
  err = _max_diff(_pts(rigged), _pts(name))
  assert err < 1e-2, "rig deviates from Anny by %.4f cm" % err
  print("[ok] rig (104 joints, skinCluster, 52 ARKit shapes) matches the "
        "panel mesh: max |err| = %.2e cm" % err)
  mc.setAttr(bs + ".jawOpen", 1.0)
  body.set_face("jawOpen", 1.0)
  err = _max_diff(_pts(rigged), _pts(name))
  assert err < 1e-2, "blendShape jawOpen deviates by %.4f cm" % err
  for j in joints[1:]:
    mc.setAttr(j + ".rotate", 0, 0, 0)
  print("[ok] blendShape weights drive the rig like the panel "
        "(max |err| = %.2e cm)" % err)

  # --- FBX ------------------------------------------------------------------
  from anny_maya.scene import export_fbx
  path = os.path.join(tempfile.gettempdir(), "anny_smoke.fbx")
  if os.path.isfile(path):
    os.remove(path)
  try:
    export_fbx.export_rigged_fbx(rigged, path=path)
    assert os.path.getsize(path) > 100000, "FBX too small"
    print("[ok] FBX export: %.1f MB" % (os.path.getsize(path) / 1e6))
    os.remove(path)
  except RuntimeError as e:
    if "fbxmaya" in str(e).lower() or "plug" in str(e).lower():
      print("[skip] FBX plugin unavailable: %s" % e)
    else:
      raise

  _head_checks(body, name)

  from anny_maya.ui import licenses
  titles = [t for t, _ in licenses.collect()]
  assert any("CC0" in t for t in titles) and any("Anny" in t for t in titles)
  print("[ok] licenses: %d documents" % len(titles))

  # --- the real panel, headless ---------------------------------------------
  try:
    try:
      from PySide6 import QtWidgets
    except ImportError:
      from PySide2 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if not isinstance(app, QtWidgets.QApplication):
      raise ImportError("no GUI QApplication")
    from anny_maya.ui import panel as ui
    body.reset_all()
    p = ui.AnnyPanel(parent=None, adopt_transform=name)
    assert p.tabs.count() == 6, "panel tabs: %d" % p.tabs.count()
    p._randomize_body()
    assert p.body.state["phenotypes"] != {k: 0.5 for k in
                                          p.body.state["phenotypes"]}
    p._phen_sliders["height"].spin.setValue(0.9)
    assert abs(p.body.state["phenotypes"]["height"] - 0.9) < 1e-9
    p.filter_edit.setText("nose")
    shown = [e for e in p._local_frames if not e["frame"].isHidden()]
    assert shown and all(e["built"] for e in shown)
    p._local_sliders["nose-scale-vert-incr"].spin.setValue(0.4)
    assert p.body.state["local_changes"]["nose-scale-vert-incr"] == 0.4
    p.filter_edit.setText("")
    p.expr_combo.setCurrentText("Smile")
    p._apply_expression()
    assert p.body.state["facial_actions"]["mouthSmileLeft"] > 0
    p._face_sliders["browInnerUp"].spin.setValue(0.5)
    p._pose_preset("tpose")
    assert p.body.state["rotations"]
    item = p._pose_items["head"]
    p.bone_tree.setCurrentItem(item)
    p._pose_sliders[1].spin.setValue(25.0)
    assert p.body.state["rotations"]["head"][1] == 25.0
    n_hist = len(p._hist)
    p._hist_step(-1)
    assert p.body.state["rotations"].get("head", [0, 0, 0])[1] != 25.0
    p._hist_step(+1)
    assert p.body.state["rotations"]["head"][1] == 25.0
    p._reset_all()
    assert not p.body.state["rotations"] and not p.body.state["facial_actions"]
    if p.body.head_topology is not None:  # Head tab (GNM)
      p.gnm_chk.setChecked(True)
      assert p.body.has_gnm_head and not p._face_frames[0].isEnabled()
      p._gnm_sliders[("identity", 0)].spin.setValue(1.5)
      assert p.body.state["head"]["identity"][0] == 1.5
      p._gnm_sliders[("expression", 0)].spin.setValue(1.0)
      assert p.body.state["head"]["expression"][100] == 1.0,           "symmetry did not mirror the left/right eye expression mode"
      p._gnm_fit_sliders["scale"].spin.setValue(1.1)
      p._randomize_gnm("identity")
      p.gnm_chk.setChecked(False)
      assert not p.body.has_gnm_head and p._face_frames[0].isEnabled()
      print("[ok] Head tab: GNM toggle, identity/expression/fit sliders, "
            "symmetry, randomize")
    p.close()
    p.deleteLater()
    print("[ok] AnnyPanel builds headlessly (6 tabs; randomize, filter, "
          "expression, pose preset, bone sliders, history %d steps, reset)"
          % n_hist)
  except ImportError:
    print("[skip] Qt not available: panel construction not exercised")

  worker.shutdown_worker()
  print("SMOKE TEST PASSED")


def _boundary_vertices(name):
  import maya.api.OpenMaya as om2
  sel = om2.MSelectionList()
  sel.add(name)
  it = om2.MItMeshEdge(sel.getDagPath(0))
  out = set()
  while not it.isDone():
    if it.onBoundary():
      out.add(it.vertexId(0))
      out.add(it.vertexId(1))
    it.next()
  return out


def _head_checks(body, name):
  """Phase 2: GNM head on the Anny body (skipped without GNM data)."""
  import json
  import time
  from maya import cmds as mc
  from anny_maya.services import bootstrap
  from anny_maya.scene import rig
  if not bootstrap.gnm_available() or body.head_topology is None:
    print("[skip] GNM head data not installed (build_module.py --gnm)")
    return
  body.reset_all()
  body.reset_pose()
  t = time.time()
  body.set_gnm_head(True)
  ht = body.head_topology
  nb = ht.meta["head"]["body_vertices"]
  assert mc.polyEvaluate(name, vertex=True) == ht.num_vertices
  print("[ok] GNM head on: %d verts (%d body + %d GNM) in %.2fs"
        % (ht.num_vertices, nb, ht.num_vertices - nb, time.time() - t))
  border = _boundary_vertices(name)
  assert not any(v < nb for v in border),       "the body has open borders: the neck seam is not closed"
  print("[ok] neck seam closed: no open border on the body side "
        "(%d border verts inside GNM's eyes/mouth)" % len(border))
  before = _pts(name)
  body.randomize_gnm("identity", seed=4)
  after = _pts(name)
  body_moved = max(abs(a - b) for a, b in zip(before[:3 * nb], after[:3 * nb]))
  head_moved = max(abs(a - b) for a, b in zip(before[3 * nb:], after[3 * nb:]))
  assert body_moved < 1e-6 and head_moved > 0.1, (body_moved, head_moved)
  print("[ok] GNM identity moves the head (%.2f cm), body untouched" %
        head_moved)
  body.randomize_gnm("expression", scale=0.5, seed=2)
  body.set_rotation("head", 1, 35.0)
  body.set_rotation("neck02", 0, 15.0)
  body.set_gnm_fit("scale", 1.05)
  rigged = rig.bake_rig(body, modes=1)
  info = rig.rig_info(rigged)
  aliases = mc.aliasAttr(info["blendShape"], query=True)[::2]
  assert "gnmExpression" in aliases and "lower_face_region_000" in aliases
  err = _max_diff(_pts(rigged), _pts(name))
  assert err < 1e-2, "GNM-head rig deviates by %.4f cm" % err
  print("[ok] GNM-head rig (one skinCluster, %d face targets) matches the "
        "panel: max |err| = %.2e cm" % (len(aliases), err))
  fake = mc.createNode("transform", name="gnm_head_fake")
  mc.addAttr(fake, longName="gnmState", dataType="string")
  ident = [0.0] * 253
  ident[0] = 2.0
  mc.setAttr(fake + ".gnmState", json.dumps(
      {"identity": ident, "expression": [0.0] * 383,
       "rotations": [[0, 0, 0]] * 4, "translation": [0, 0, 0]}),
      type="string")
  body.import_gnm_head(fake)
  assert body.state["head"]["identity"][0] == 2.0
  print("[ok] import from a gnm-maya head (gnmState)")
  body.set_gnm_head(False)
  assert mc.polyEvaluate(name, vertex=True) == body.body_topology.num_vertices
  print("[ok] GNM head off: back to Anny's own head")


if __name__ == "__main__":
  try:
    run_headless()
  except Exception as e:
    import traceback
    traceback.print_exc()
    sys.stderr.write("SMOKE TEST FAILED: %s\n" % e)
    sys.exit(1)
