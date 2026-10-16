"""In-Maya GUI smoke test for the Anny panel (needs the full Maya app).

Run from the Anny menu (Run GUI Test) or the Script Editor's Python tab:

    exec(open(r"<...>/anny-maya/tests/gui_smoke_test.py").read())

Opens the real panel through anny_maya.show_ui(), drives every tab, prints
PASS/FAIL per check, then deletes what it created (the panel stays open on
a fresh body for hand inspection). Sets RESULT = (passed, failed).
"""

from maya import cmds as mc

import anny_maya

_PASS, _FAIL = [], []


def _check(label, fn):
  try:
    fn()
    _PASS.append(label)
    print("[PASS] %s" % label)
  except Exception as e:  # keep going: report every failure
    _FAIL.append(label)
    print("[FAIL] %s: %s" % (label, e))


def _main():
  before = set(mc.ls(assemblies=True) or [])
  panel = anny_maya.show_ui()
  assert panel is not None, "panel did not open"
  body = panel.body

  def phen():
    panel._phen_sliders["age"].spin.setValue(0.9)
    assert abs(body.state["phenotypes"]["age"] - 0.9) < 1e-9
    panel._randomize_body()

  def local():
    panel.filter_edit.setText("waist")
    hit = [l for l in panel._local_sliders if "waist" in l]
    assert hit, "filter built no waist sliders"
    panel._local_sliders[hit[0]].spin.setValue(0.5)
    assert body.state["local_changes"][hit[0]] == 0.5
    panel.filter_edit.setText("")
    panel._randomize_local()

  def face():
    panel.expr_combo.setCurrentText("Surprise")
    panel._apply_expression()
    assert body.state["facial_actions"]["browInnerUp"] > 0
    panel._face_sliders["eyeBlinkLeft"].spin.setValue(1.0)
    assert body.state["facial_actions"].get("eyeBlinkRight") == 1.0, \
        "symmetry did not mirror the blink"

  def pose():
    panel._pose_preset("tpose")
    panel.bone_tree.setCurrentItem(panel._pose_items["neck01"])
    panel._pose_sliders[0].spin.setValue(15.0)
    assert body.state["rotations"]["neck01"][0] == 15.0

  def history():
    n = panel._hist_i
    panel._hist_step(-1)
    assert panel._hist_i == n - 1
    panel._hist_step(+1)
    assert panel._hist_i == n

  def bake():
    panel._bake_rig()
    assert getattr(panel, "_last_rig", None) and mc.objExists(panel._last_rig)

  _check("Create: phenotype slider + randomize", phen)
  _check("Body: filter, slider, randomize", local)
  _check("Face: expression preset + symmetric slider", face)
  _check("Pose: T-pose preset + bone slider", pose)
  _check("History back/forward", history)
  _check("Animate: Bake Rig", bake)

  def head():
    from anny_maya.services import bootstrap
    if not bootstrap.gnm_available():
      print("  (GNM data not installed - skipping the head check)")
      return
    panel.gnm_chk.setChecked(True)
    assert panel.body.has_gnm_head, "GNM head did not switch on"
    panel._randomize_gnm("identity")
    panel._bake_rig()
    panel.gnm_chk.setChecked(False)
    assert not panel.body.has_gnm_head

  _check("Head: GNM head on/randomize/bake/off", head)
  _check("Licenses dialog", lambda: anny_maya.show_licenses().close())

  # cleanup: everything this test created, then a fresh body in the panel
  for node in set(mc.ls(assemblies=True) or []) - before:
    if mc.objExists(node):
      mc.delete(node)
  panel._new_body()
  print("GUI SMOKE TEST: %d passed, %d failed" % (len(_PASS), len(_FAIL)))
  return len(_PASS), len(_FAIL)


RESULT = _main()
