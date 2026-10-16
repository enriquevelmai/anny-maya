"""Tabbed Anny control panel: Create / Body / Face / Pose / Animate.

Native Maya look (plain Qt widgets, Maya's own palette, no custom styling):
labeled horizontal sliders in collapsible frames, short titled tooltips,
power features collapsed. Drives a live AnnyBody through the resident worker.
Works with PySide6 (Maya 2025+) or PySide2 (Maya 2022-2024).
"""

from __future__ import annotations

import logging
import random
import re

try:
  from PySide6 import QtWidgets, QtCore, QtGui
  from shiboken6 import wrapInstance
except ImportError:  # Maya 2022-2024
  from PySide2 import QtWidgets, QtCore, QtGui
  from shiboken2 import wrapInstance

from maya import OpenMayaUI as omui
from maya import cmds as mc

from anny_maya.core.body import AnnyBody, AGE_MIN, find_bodies
from anny_maya.ui import icons
from anny_maya.ui.widgets import CollapsibleFrame, ParamSlider

logger = logging.getLogger(__name__)

_WINDOW = None
_OBJECT_NAME = "annyBodyPanel"
_TITLE = "Anny Body"

# (label, key, lo, hi, tooltip)
_MAIN_PHENOTYPES = [
    ("Gender", "gender", 0.0, 1.0,
     "<b>Gender</b><br>0 = male, 1 = female (blends in between)."),
    ("Age", "age", AGE_MIN, 1.0,
     "<b>Age</b><br>-0.33 newborn, 0 baby, 0.33 child, 0.67 young adult, "
     "1 old."),
    ("Muscle", "muscle", 0.0, 1.0, "<b>Muscle</b><br>0 = min, 1 = max."),
    ("Weight", "weight", 0.0, 1.0, "<b>Weight</b><br>0 = min, 1 = max."),
    ("Height", "height", 0.0, 1.0, "<b>Height</b><br>0 = min, 1 = max."),
    ("Proportions", "proportions", 0.0, 1.0,
     "<b>Proportions</b><br>0 = ideal, 1 = uncommon body proportions."),
]
_MORE_PHENOTYPES = [
    ("Cup size", "cupsize", 0.0, 1.0, "<b>Cup size</b><br>Breast size."),
    ("Firmness", "firmness", 0.0, 1.0, "<b>Firmness</b><br>Breast firmness."),
    ("African", "african", 0.0, 1.0,
     "<b>African</b><br>Ancestry blend weight (the three are normalized)."),
    ("Asian", "asian", 0.0, 1.0,
     "<b>Asian</b><br>Ancestry blend weight (the three are normalized)."),
    ("Caucasian", "caucasian", 0.0, 1.0,
     "<b>Caucasian</b><br>Ancestry blend weight (the three are normalized)."),
]

# Body-tab region order (MakeHuman target folders) and display names.
_REGIONS = [("head", "Head"), ("forehead", "Forehead"),
            ("eyebrows", "Eyebrows"), ("eyes", "Eyes"), ("ears", "Ears"),
            ("nose", "Nose"), ("cheek", "Cheeks"), ("mouth", "Mouth"),
            ("chin", "Chin"), ("neck", "Neck"), ("torso", "Torso"),
            ("breast", "Breast"), ("stomach", "Stomach"), ("hip", "Hip"),
            ("pelvis", "Pelvis"), ("buttocks", "Buttocks"), ("arms", "Arms"),
            ("hands", "Hands"), ("legs", "Legs"), ("feet", "Feet"),
            ("measure", "Measurements")]

# A few quick expressions built from the ARKit units (Face tab).
_EXPRESSIONS = {
    "Smile": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8,
              "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4,
              "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2},
    "Surprise": {"browInnerUp": 0.8, "browOuterUpLeft": 0.7,
                 "browOuterUpRight": 0.7, "eyeWideLeft": 0.6,
                 "eyeWideRight": 0.6, "jawOpen": 0.35},
    "Anger": {"browDownLeft": 0.8, "browDownRight": 0.8,
              "noseSneerLeft": 0.5, "noseSneerRight": 0.5,
              "mouthPressLeft": 0.4, "mouthPressRight": 0.4,
              "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3},
    "Sadness": {"browInnerUp": 0.6, "mouthFrownLeft": 0.6,
                "mouthFrownRight": 0.6, "mouthShrugLower": 0.3},
    "Blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
    "Open mouth": {"jawOpen": 0.6, "mouthLowerDownLeft": 0.3,
                   "mouthLowerDownRight": 0.3},
}

INFO = (
    "Anny is NAVER's parametric human body model (MakeHuman-based, CC0 "
    "assets): one mesh and one 104-bone skeleton for every age, gender and "
    "build.\n\n"
    "• Create — phenotypes (gender, age, muscle, weight, height, "
    "proportions). Randomize draws a plausible body.\n"
    "• Body — 250+ local MakeHuman changes by region (-1..1).\n"
    "• Face — the 52 ARKit facial actions (0..1), on Anny's head or on "
    "the GNM head.\n"
    "• Head — swap in a Google GNM head (identity + expression), stitched "
    "at the neck and skinned to the same skeleton; import gnm-maya faces.\n"
    "• Pose — per-bone rotations about Maya's world axes at the neutral "
    "pose (Anny's 'local-ref' parameterization) + presets.\n"
    "• Animate — Bake Rig turns the body into joints + skinCluster + 52 "
    "ARKit blendShapes (no Anny runtime needed) and exports FBX.\n\n"
    "Symmetry mirrors L/R edits in Body, Face and Pose. ◀ ▶ step through "
    "the history of looks. Double-click a slider to reset it.")


def maya_main_window():
  ptr = omui.MQtUtil.mainWindow()
  return wrapInstance(int(ptr), QtWidgets.QWidget) if ptr else None


def _camel_words(name):
  return " ".join(w.capitalize() for w in
                  re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name))


def _local_title(pos, neg):
  """'l-lowerarm-scale-depth-incr' / '...-decr' -> 'L lowerarm scale depth'."""
  a, b = pos.split("-"), neg.split("-")
  common = [x for x, y in zip(a, b) if x == y]
  words = common or a
  if words and words[0] in ("l", "r"):
    words = [words[0].upper()] + words[1:]
  title = " ".join(words)
  return title[:1].upper() + title[1:]


class AnnyPanel(QtWidgets.QWidget):

  def __init__(self, parent=None, adopt_transform=None):
    super(AnnyPanel, self).__init__(parent)
    self.setObjectName(_OBJECT_NAME)
    self.setWindowFlags(QtCore.Qt.Window)
    self.setWindowTitle(_TITLE)
    self.setWindowIcon(icons.window_icon())
    self.setMinimumSize(460, 520)
    self.resize(540, 760)

    self.body = None
    self._phen_sliders = {}
    self._local_sliders = {}
    self._local_frames = []       # (region, frame, labels, built?)
    self._face_sliders = {}
    self._face_frames = []
    self._gnm_sliders = {}        # (kind, index) -> ParamSlider
    self._gnm_fit_sliders = {}    # key -> ParamSlider
    self._head_built = False
    self._pose_sliders = []       # 3 axis sliders for the selected bone
    self._trans_sliders = []
    self._pose_items = {}         # bone -> QTreeWidgetItem
    self._bone = None             # currently selected bone
    self._hist = []
    self._hist_i = -1
    self._hist_max = 30
    self._last_error = None

    self._refresh_timer = QtCore.QTimer(self)
    self._refresh_timer.setSingleShot(True)
    self._refresh_timer.setInterval(25)
    self._refresh_timer.timeout.connect(self._do_refresh)

    self.status = QtWidgets.QLabel("Building body…")
    self.status.setWordWrap(True)
    self.sym_chk = QtWidgets.QCheckBox("Symmetry")
    self.sym_chk.setChecked(True)
    self.sym_chk.setToolTip(
        "<b>Symmetry</b><br>Mirror left/right edits in the Body, Face and "
        "Pose tabs so both sides stay matched.")
    self.tabs = QtWidgets.QTabWidget()

    try:
      if adopt_transform and mc.objExists(adopt_transform):
        self.body = AnnyBody.adopt(adopt_transform)
        self.status.setText("Adopted: %s" % self.body.transform)
      else:
        self.body = AnnyBody()
        self.status.setText("Body: %s" % self.body.transform)
    except Exception as e:
      self._show_error("Failed to build the Anny body", e)

    self._build_ui()
    self._sync_from_body()
    self._push_history()

  # --- history ladder ------------------------------------------------------

  def _push_history(self):
    """Record the state after a discrete action or a finished slider drag."""
    if not self.body:
      return
    snap = self.body.get_state()
    if 0 <= self._hist_i < len(self._hist) and self._hist[self._hist_i] == snap:
      return
    del self._hist[self._hist_i + 1:]
    self._hist.append(snap)
    if len(self._hist) > self._hist_max:
      self._hist.pop(0)
    self._hist_i = len(self._hist) - 1
    self._update_hist_buttons()

  def _hist_step(self, delta):
    if not self.body or not self._hist:
      return
    j = self._hist_i + delta
    if not (0 <= j < len(self._hist)):
      return
    try:
      self._refresh_timer.stop()
      self._hist_i = j
      self.body.set_state(self._hist[j])
      self._sync_from_body()
      self.status.setText("History: state %d / %d." % (j + 1, len(self._hist)))
      self._update_hist_buttons()
    except Exception as e:
      self._show_error("History step failed", e)

  def _update_hist_buttons(self):
    if hasattr(self, "hist_back_btn"):
      self.hist_back_btn.setEnabled(self._hist_i > 0)
      self.hist_fwd_btn.setEnabled(self._hist_i < len(self._hist) - 1)

  # --- errors --------------------------------------------------------------

  def _show_error(self, context, err):
    """Log the traceback, show it in the status bar and pop a dialog (the
    same message is not re-popped back to back - a failing drag would)."""
    logger.exception(context)
    msg = str(err) or err.__class__.__name__
    self.status.setText("Error: %s" % msg)
    key = (context, msg)
    if key == self._last_error:
      return
    self._last_error = key
    QtWidgets.QMessageBox.critical(
        self, "Anny - %s" % context,
        "%s\n\n%s\n\nSee the Script Editor for the full traceback."
        % (context, msg))

  # --- layout --------------------------------------------------------------

  def _build_ui(self):
    outer = QtWidgets.QVBoxLayout(self)
    top = QtWidgets.QHBoxLayout()
    top.addWidget(self.status, 1)
    top.addWidget(self.sym_chk)
    info = QtWidgets.QPushButton()
    info.setFixedWidth(28)
    info.setToolTip("<b>About the controls</b><br>What each tab does.")
    icons.decorate(info, "info", 16)
    info.clicked.connect(lambda: QtWidgets.QMessageBox.information(
        self, "Anny Body", INFO))
    top.addWidget(info)
    outer.addLayout(top)
    outer.addWidget(self.tabs, 1)

    if self.body is not None:
      self.tabs.addTab(self._create_tab(), icons.icon("anny"), "Create")
      self.tabs.addTab(self._body_tab(), icons.icon("body"), "Body")
      self.tabs.addTab(self._face_tab(), icons.icon("face"), "Face")
      self.tabs.addTab(self._head_tab(), icons.icon("head"), "Head")
      self.tabs.addTab(self._pose_tab(), icons.icon("pose"), "Pose")
      self.tabs.addTab(self._animate_tab(), icons.icon("cube"), "Animate")

    bottom = QtWidgets.QHBoxLayout()
    self.hist_back_btn = self._icon_btn(
        "arrow_back", "<b>Previous look</b><br>Step back through the history "
        "(last %d states)." % self._hist_max, lambda: self._hist_step(-1))
    self.hist_fwd_btn = self._icon_btn(
        "arrow_forward", "<b>Next look</b><br>Step forward again.",
        lambda: self._hist_step(+1))
    bottom.addWidget(self.hist_back_btn)
    bottom.addWidget(self.hist_fwd_btn)
    bottom.addStretch(1)
    disp = QtWidgets.QPushButton("Display")
    disp.setToolTip("<b>Display</b><br>Show/hide eyes and tongue, pick the "
                    "skin colour, frame the body.")
    icons.decorate(disp, "tune", 15)
    disp.setMenu(self._display_menu())
    bottom.addWidget(disp)
    reset = QtWidgets.QPushButton("Reset All")
    reset.setToolTip("<b>Reset all</b><br>Back to the neutral average body: "
                     "phenotypes 0.5, no local changes, no expression, "
                     "neutral pose.")
    icons.decorate(reset, "restart", 15)
    reset.clicked.connect(self._reset_all)
    bottom.addWidget(reset)
    outer.addLayout(bottom)
    self._update_hist_buttons()

  def _icon_btn(self, icon, tip, slot, width=30):
    b = QtWidgets.QPushButton()
    b.setFixedWidth(width)
    b.setToolTip(tip)
    icons.decorate(b, icon, 15)
    b.clicked.connect(slot)
    return b

  def _btn(self, text, icon, tip, slot):
    b = QtWidgets.QPushButton(text)
    b.setToolTip(tip)
    if icon:
      icons.decorate(b, icon, 15)
    b.clicked.connect(slot)
    return b

  def _scroll(self, inner):
    sc = QtWidgets.QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.NoFrame)
    sc.setWidget(inner)
    return sc

  def _display_menu(self):
    from anny_maya.scene import material
    menu = QtWidgets.QMenu(self)
    for label in material.COMPONENT_GROUPS:
      act = menu.addAction(label)
      act.setCheckable(True)
      act.setChecked(True)
      act.toggled.connect(lambda on, l=label: self._set_visible(l, on))
    menu.addSeparator()
    menu.addAction("Skin Colour…", self._pick_skin_color)
    menu.addAction("Frame Body", lambda: _frame(self.body))

    def _sync():
      for act in menu.actions():
        if act.isCheckable():
          act.blockSignals(True)
          act.setChecked(material.component_visible(act.text()))
          act.blockSignals(False)
    menu.aboutToShow.connect(_sync)
    return menu

  def _set_visible(self, label, on):
    from anny_maya.scene import material
    try:
      material.set_component_visible(label, on)
      self.status.setText("%s %s." % (label, "shown" if on else "hidden"))
    except Exception as e:
      self._show_error("Display toggle failed", e)

  def _pick_skin_color(self):
    from anny_maya.scene import material
    c = QtWidgets.QColorDialog.getColor(QtGui.QColor(204, 153, 128), self,
                                        "Skin colour")
    if c.isValid():
      try:
        material.set_skin_color((c.redF(), c.greenF(), c.blueF()))
      except Exception as e:
        self._show_error("Skin colour failed", e)

  # --- Create tab ------------------------------------------------------------

  def _create_tab(self):
    host = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(host)
    v.setSpacing(6)

    row = QtWidgets.QHBoxLayout()
    row.addWidget(self._btn(
        "Randomize", "dice", "<b>Randomize body</b><br>Draw a plausible random "
        "body (adults unless 'Children' is checked).", self._randomize_body))
    self.child_chk = QtWidgets.QCheckBox("Children")
    self.child_chk.setToolTip("<b>Children</b><br>Let Randomize pick any age, "
                              "from newborn to old.")
    self.more_chk = QtWidgets.QCheckBox("All phenotypes")
    self.more_chk.setToolTip("<b>All phenotypes</b><br>Randomize also rolls "
                             "cup size, firmness and ancestry.")
    row.addWidget(self.child_chk)
    row.addWidget(self.more_chk)
    row.addStretch(1)
    row.addWidget(self._btn(
        "Reset", "restart", "<b>Reset phenotypes</b><br>All back to 0.5 "
        "(the average body).", self._reset_phenotypes))
    row.addWidget(self._btn(
        "New Body", "add_box", "<b>New body</b><br>Create another Anny body "
        "and control it from this panel (the current one stays in the "
        "scene).", self._new_body))
    v.addLayout(row)

    main = CollapsibleFrame("Phenotype", expanded=True)
    for label, key, lo, hi, tip in _MAIN_PHENOTYPES:
      main.content_layout().addWidget(self._phen_slider(label, key, lo, hi, tip))
    v.addWidget(main)
    more = CollapsibleFrame("More phenotypes", expanded=False)
    for label, key, lo, hi, tip in _MORE_PHENOTYPES:
      more.content_layout().addWidget(self._phen_slider(label, key, lo, hi, tip))
    v.addWidget(more)
    v.addStretch(1)
    return self._scroll(host)

  def _phen_slider(self, label, key, lo, hi, tip):
    w = ParamSlider(label, lo, hi, 0.5,
                    lambda val, k=key: self._on_phenotype(k, val),
                    tooltip=tip, label_width=90,
                    on_release=self._push_history)
    self._phen_sliders[key] = w
    return w

  # --- Body tab --------------------------------------------------------------

  def _body_tab(self):
    container = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(container)
    row = QtWidgets.QHBoxLayout()
    self.filter_edit = QtWidgets.QLineEdit()
    self.filter_edit.setPlaceholderText("Filter (e.g. nose, waist, l-)…")
    self.filter_edit.setClearButtonEnabled(True)
    self.filter_edit.setToolTip("<b>Filter</b><br>Show only the changes whose "
                                "name contains this text.")
    self.filter_edit.textChanged.connect(self._apply_filter)
    row.addWidget(self.filter_edit, 1)
    self.local_amount = QtWidgets.QDoubleSpinBox()
    self.local_amount.setRange(0.0, 1.0)
    self.local_amount.setSingleStep(0.05)
    self.local_amount.setValue(0.25)
    self.local_amount.setToolTip("<b>Amount</b><br>Strength of Randomize "
                                 "(+- this value).")
    row.addWidget(self._btn(
        "Randomize", "dice", "<b>Randomize local changes</b><br>Random values "
        "for every change in the EXPANDED regions (all regions if none is "
        "expanded), scaled by Amount.", self._randomize_local))
    row.addWidget(self.local_amount)
    row.addWidget(self._btn(
        "Reset", "restart", "<b>Reset local changes</b><br>Zero every local "
        "change.", self._reset_local))
    v.addLayout(row)

    host = QtWidgets.QWidget()
    hv = QtWidgets.QVBoxLayout(host)
    hv.setSpacing(2)
    meta = self.body.topology.meta
    regions = meta["local_regions"]
    labels = meta["local_labels"]
    negs = dict(zip(labels, meta["local_neg_labels"]))
    by_region = {}
    for lab, reg in zip(labels, regions):
      by_region.setdefault(reg, []).append(lab)
    order = [r for r, _ in _REGIONS if r in by_region]
    order += [r for r in by_region if r not in order]
    names = dict(_REGIONS)
    for reg in order:
      labs = by_region[reg]
      frame = CollapsibleFrame("%s  (%d)" % (names.get(reg, reg.title()),
                                             len(labs)), expanded=False)
      entry = {"region": reg, "frame": frame, "labels": labs,
               "negs": negs, "built": False}
      frame.toggled.connect(lambda on, e=entry: on and self._build_region(e))
      self._local_frames.append(entry)
      hv.addWidget(frame)
    hv.addStretch(1)
    v.addWidget(self._scroll(host), 1)
    return container

  def _build_region(self, entry):
    if entry["built"]:
      return
    entry["built"] = True
    lay = entry["frame"].content_layout()
    btns = QtWidgets.QHBoxLayout()
    btns.addStretch(1)
    btns.addWidget(self._btn(
        "Randomize region", "dice", "<b>Randomize region</b><br>Random values "
        "for this region only.",
        lambda _=False, e=entry: self._randomize_local(e["labels"])))
    btns.addWidget(self._btn(
        "Reset region", "restart", "<b>Reset region</b><br>Zero this region.",
        lambda _=False, e=entry: self._reset_local(e["labels"])))
    lay.addLayout(btns)
    for lab in entry["labels"]:
      neg = entry["negs"][lab]
      tip = "<b>%s</b><br>-1: %s<br>+1: %s" % (_local_title(lab, neg), neg, lab)
      w = ParamSlider(_local_title(lab, neg), -1.0, 1.0, 0.0,
                      lambda val, l=lab: self._on_local(l, val), tooltip=tip,
                      label_width=190, on_release=self._push_history)
      w.setProperty("anny_label", lab)
      w.setProperty("anny_search", (lab + " " + neg).lower())
      self._local_sliders[lab] = w
      lay.addWidget(w)
    self._sync_local()
    self._apply_filter(self.filter_edit.text())

  def _apply_filter(self, text):
    text = (text or "").strip().lower()
    for e in self._local_frames:
      if text:
        hits = [l for l in e["labels"]
                if text in (l + " " + e["negs"][l]).lower()]
        e["frame"].setVisible(bool(hits))
        if hits:
          self._build_region(e)
          e["frame"].set_expanded(True)
      else:
        e["frame"].setVisible(True)
      for lab in e["labels"]:
        w = self._local_sliders.get(lab)
        if w is not None:
          w.setVisible(not text or text in w.property("anny_search"))

  # --- Face tab ----------------------------------------------------------------

  def _face_tab(self):
    host = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(host)
    v.setSpacing(4)
    row = QtWidgets.QHBoxLayout()
    self.expr_combo = QtWidgets.QComboBox()
    self.expr_combo.addItems(sorted(_EXPRESSIONS))
    self.expr_combo.setToolTip("<b>Quick expression</b><br>A preset built "
                               "from the ARKit units.")
    row.addWidget(self.expr_combo)
    row.addWidget(self._btn("Apply", "mood", "<b>Apply expression</b><br>"
                            "Replace the face with the chosen preset.",
                            self._apply_expression))
    row.addWidget(self._btn("Randomize", "dice", "<b>Random expression</b>"
                            "<br>A few random face units.",
                            self._randomize_face))
    row.addStretch(1)
    row.addWidget(self._btn("Reset", "restart", "<b>Reset face</b><br>Neutral "
                            "expression (all units 0).", self._reset_face))
    v.addLayout(row)
    self.face_note = QtWidgets.QLabel(
        "The ARKit actions on the GNM head need a GNM data update: in the "
        "Head tab, turn Use GNM head off and on again.")
    self.face_note.setWordWrap(True)
    self.face_note.setVisible(False)
    v.addWidget(self.face_note)
    self._face_row = row
    for title, labels in self.body.topology.meta["face_groups"]:
      frame = CollapsibleFrame(title, expanded=title in ("Mouth", "Eyes"))
      self._face_frames.append(frame)
      for lab in labels:
        w = ParamSlider(_camel_words(lab), 0.0, 1.0, 0.0,
                        lambda val, l=lab: self._on_face(l, val),
                        tooltip="<b>%s</b><br>ARKit facial action (0..1)." % lab,
                        label_width=170, on_release=self._push_history)
        self._face_sliders[lab] = w
        frame.content_layout().addWidget(w)
      v.addWidget(frame)
    v.addStretch(1)
    return self._scroll(host)

  # --- Head tab (Google GNM head, Phase 2) ------------------------------------------

  def _head_tab(self):
    host = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(host)
    v.setSpacing(6)
    self._head_layout = v
    row = QtWidgets.QHBoxLayout()
    self.gnm_chk = QtWidgets.QCheckBox("Use GNM head")
    self.gnm_chk.setToolTip(
        "<b>GNM head</b><br>Replace Anny's head with a Google GNM head "
        "(253 identity + 383 expression modes), stitched at the neck and "
        "skinned to Anny's skeleton. Downloads the GNM model (~40 MB) the "
        "first time.")
    self.gnm_chk.toggled.connect(self._on_gnm_toggled)
    row.addWidget(self.gnm_chk)
    row.addStretch(1)
    imp = self._btn("Import from gnm-maya", "download",
                    "<b>Import face</b><br>Copy identity + expression from "
                    "a gnm-maya head in the scene (select it first if there "
                    "are several).", self._import_gnm)
    row.addWidget(imp)
    v.addLayout(row)
    self.head_note = QtWidgets.QLabel(
        "GNM's sliders are a statistical basis ordered by importance (m0 = "
        "broadest change), like gnm-maya. The first modes of each region are "
        "shown; all are scriptable (body.set_gnm).")
    self.head_note.setWordWrap(True)
    v.addWidget(self.head_note)
    self._head_controls = QtWidgets.QWidget()
    self._head_controls_layout = QtWidgets.QVBoxLayout(self._head_controls)
    self._head_controls_layout.setContentsMargins(0, 0, 0, 0)
    v.addWidget(self._head_controls)
    v.addStretch(1)
    self._build_head_controls()
    return self._scroll(host)

  def _build_head_controls(self):
    """Sliders need the GNM metadata, known once the data is installed."""
    if self._head_built or not self.body or not self.body.head_topology:
      return
    self._head_built = True
    lay = self._head_controls_layout
    self.gnm_scale = {}
    for kind, title, groups, shown, scale in (
        ("identity", "Identity", self.body.gnm_identity_groups, 12, 1.0),
        ("expression", "Expression", self.body.gnm_expression_groups, 8,
         0.5)):
      frame = CollapsibleFrame(title, expanded=True)
      row = QtWidgets.QHBoxLayout()
      row.addWidget(self._btn(
          "Randomize", "dice", "<b>Randomize %s</b><br>Gaussian values for "
          "every %s mode, times Amount." % (kind, kind),
          lambda _=False, k=kind: self._randomize_gnm(k)))
      spin = QtWidgets.QDoubleSpinBox()
      spin.setRange(0.0, 3.0)
      spin.setSingleStep(0.1)
      spin.setValue(scale)
      spin.setToolTip("<b>Amount</b><br>Standard deviations for Randomize.")
      self.gnm_scale[kind] = spin
      row.addWidget(spin)
      row.addStretch(1)
      row.addWidget(self._btn(
          "Reset", "restart", "<b>Reset %s</b><br>All %s modes to 0 (the "
          "average)." % (kind, kind),
          lambda _=False, k=kind: self._reset_gnm(k)))
      frame.content_layout().addLayout(row)
      for label, start, end in groups:
        n = min(shown, end - start + 1)
        sub = CollapsibleFrame("%s  (%d of %d)" % (label.replace("_", " "),
                                                   n, end - start + 1),
                               expanded=label in ("head", "lower_face_region"))
        for k in range(n):
          i = start + k
          w = ParamSlider("m%d" % k, -3.0, 3.0, 0.0,
                          lambda val, kd=kind, ii=i: self._on_gnm(kd, ii, val),
                          tooltip="<b>%s m%d</b><br>GNM %s mode #%d (-3..3 "
                          "standard deviations)." % (label, k, kind, i),
                          label_width=40, on_release=self._push_history)
          self._gnm_sliders[(kind, i)] = w
          sub.content_layout().addWidget(w)
        frame.content_layout().addWidget(sub)
      lay.addWidget(frame)
    fit = CollapsibleFrame("Fit to the body", expanded=False)
    for label, key, lo, hi, dflt, tip in (
        ("Scale", "scale", 0.8, 1.25, 1.0,
         "<b>Scale</b><br>Head size relative to the eye-distance fit."),
        ("Up/Down", ("offset", 1), -5.0, 5.0, 0.0,
         "<b>Up/Down</b><br>Move the head along the neck (cm)."),
        ("Fwd/Back", ("offset", 2), -5.0, 5.0, 0.0,
         "<b>Forward/Back</b><br>Move the head forward/back (cm)."),
        ("Pitch", "pitch", -20.0, 20.0, 0.0,
         "<b>Pitch</b><br>Tilt the head about the eye axis (degrees).")):
      w = ParamSlider(label, lo, hi, dflt,
                      lambda val, k=key: self._on_gnm_fit(k, val),
                      tooltip=tip, label_width=70,
                      on_release=self._push_history)
      self._gnm_fit_sliders[key] = w
      fit.content_layout().addWidget(w)
    lay.addWidget(fit)

  def _sync_head(self):
    on = bool(self.body and self.body.has_gnm_head)
    self.gnm_chk.blockSignals(True)
    self.gnm_chk.setChecked(on)
    self.gnm_chk.blockSignals(False)
    self._head_controls.setEnabled(on)
    face_ok = not on or self.body.gnm_arkit
    self.face_note.setVisible(not face_ok)
    for f in self._face_frames:
      f.setEnabled(face_ok)
    for i in range(self._face_row.count()):
      w = self._face_row.itemAt(i).widget()
      if w is not None:
        w.setEnabled(face_ok)
    if hasattr(self, "bake_modes_spin"):
      self.bake_modes_spin.setEnabled(on)
      self.bake_face_chk.setText("52 ARKit + GNM expression" if on
                                 else "52 ARKit face shapes")
    if not on:
      return
    h = self.body.state["head"]
    for (kind, i), w in self._gnm_sliders.items():
      w.set_value_silent(h[kind][i])
    for key, w in self._gnm_fit_sliders.items():
      fit = h["fit"]
      w.set_value_silent(fit[key[0]][key[1]] if isinstance(key, tuple)
                         else fit[key])

  def _on_gnm_toggled(self, on):
    from anny_maya.services import bootstrap
    try:
      self._refresh_timer.stop()
      if on and not bootstrap.gnm_ready():
        if not bootstrap.ensure_gnm_with_dialog():
          self._sync_head()
          return
      if on and (self.body.head_topology is None
                 or not self.body.gnm_arkit):
        # The worker started before GNM was installed: restart it.
        from anny_maya.core import worker as _worker
        _worker.shutdown_worker()
        self.body = AnnyBody.adopt(self.body.transform)
        self._build_head_controls()
      self.body.set_gnm_head(on)
      self._sync_from_body()
      self._push_history()
      self.status.setText("GNM head %s." % ("on" if on else "off - Anny's "
                                             "own head"))
    except Exception as e:
      self._sync_head()
      self._show_error("GNM head toggle failed", e)

  def _on_gnm(self, kind, i, value):
    try:
      changed = self.body.set_gnm(kind, i, value,
                                  symmetry=self.sym_chk.isChecked(),
                                  update=False)
      for j in changed:
        if j != i and (kind, j) in self._gnm_sliders:
          self._gnm_sliders[(kind, j)].set_value_silent(value)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("GNM edit failed", e)

  def _on_gnm_fit(self, key, value):
    try:
      self.body.set_gnm_fit(key, value, update=False)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Head fit failed", e)

  def _randomize_gnm(self, kind):
    self._run("Randomize failed", lambda: self.body.randomize_gnm(
        kind, scale=self.gnm_scale[kind].value(),
        seed=random.randint(0, 1 << 30),
        symmetric=self.sym_chk.isChecked()), msg="Randomized GNM %s." % kind)

  def _reset_gnm(self, kind):
    self._run("Reset failed", lambda: self.body.reset_gnm(kind),
              msg="GNM %s reset." % kind)

  def _import_gnm(self):
    from anny_maya.core.body import find_gnm_heads
    heads = find_gnm_heads(selected_only=True) or find_gnm_heads()
    if not heads:
      self._show_error("Import from gnm-maya",
                       RuntimeError("No gnm-maya head in the scene."))
      return
    if not self.body.has_gnm_head:
      self.gnm_chk.setChecked(True)
      if not self.body.has_gnm_head:
        return
    result = {}

    def go():
      result["sculpt"] = self.body.import_gnm_head(heads[0])
    self._run("Import failed", go, msg="Imported the face of %s%s." % (
        heads[0], ""))
    if result.get("sculpt"):
      self.status.setText("Imported %s (its masked sculpt layer is not "
                          "transferable - coefficients only)." % heads[0])

  # --- Pose tab ----------------------------------------------------------------

  def _pose_tab(self):
    container = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(container)
    row = QtWidgets.QHBoxLayout()
    for text, name, tip in (
        ("Neutral", "neutral", "<b>Neutral pose</b><br>Anny's reference "
         "A-pose (all rotations 0)."),
        ("T-Pose", "tpose", "<b>T-pose</b><br>Arms straight out, aimed for "
         "this body's proportions."),
        ("Arms Down", "arms_down", "<b>Arms down</b><br>Relaxed arms along "
         "the body.")):
      row.addWidget(self._btn(text, "pose", tip,
                              lambda _=False, n=name: self._pose_preset(n)))
    row.addStretch(1)
    row.addWidget(self._btn("Reset", "restart", "<b>Reset pose</b><br>All "
                            "rotations and the root position back to 0.",
                            self._reset_pose))
    v.addLayout(row)

    split = QtWidgets.QHBoxLayout()
    self.bone_tree = QtWidgets.QTreeWidget()
    self.bone_tree.setHeaderHidden(True)
    self.bone_tree.setMinimumWidth(170)
    self.bone_tree.setToolTip("<b>Bones</b><br>Pick a bone to rotate. "
                              "Posed bones are shown in bold.")
    for title, bones in self.body.topology.meta["bone_groups"]:
      top = QtWidgets.QTreeWidgetItem([title])
      top.setFlags(top.flags() & ~QtCore.Qt.ItemIsSelectable)
      self.bone_tree.addTopLevelItem(top)
      for b in bones:
        it = QtWidgets.QTreeWidgetItem([b])
        it.setData(0, QtCore.Qt.UserRole, b)
        top.addChild(it)
        self._pose_items[b] = it
      top.setExpanded(title in ("Spine & Head", "Left Arm", "Right Arm"))
    self.bone_tree.currentItemChanged.connect(self._on_bone_selected)
    split.addWidget(self.bone_tree, 2)

    right = QtWidgets.QVBoxLayout()
    self.bone_label = QtWidgets.QLabel("Select a bone")
    f = self.bone_label.font()
    f.setBold(True)
    self.bone_label.setFont(f)
    right.addWidget(self.bone_label)
    for axis, name in enumerate(("Rotate X", "Rotate Y", "Rotate Z")):
      w = ParamSlider(name, -180.0, 180.0, 0.0,
                      lambda val, a=axis: self._on_rotation(a, val),
                      tooltip="<b>%s</b><br>Degrees about Maya's world %s axis "
                      "at the neutral pose." % (name, "XYZ"[axis]),
                      decimals=1, label_width=60,
                      on_release=self._push_history)
      w.setEnabled(False)
      self._pose_sliders.append(w)
      right.addWidget(w)
    brow = QtWidgets.QHBoxLayout()
    brow.addWidget(self._btn("Reset Bone", "restart", "<b>Reset bone</b><br>"
                             "Zero the selected bone's rotation.",
                             self._reset_bone))
    brow.addWidget(self._btn("Mirror L→R", "shuffle", "<b>Mirror left to "
                             "right</b><br>Copy every left bone rotation onto "
                             "the right side.",
                             lambda: self._mirror_pose("L")))
    brow.addWidget(self._btn("Mirror R→L", "shuffle", "<b>Mirror right to "
                             "left</b><br>Copy every right bone rotation onto "
                             "the left side.",
                             lambda: self._mirror_pose("R")))
    right.addLayout(brow)
    root = CollapsibleFrame("Root position (cm)", expanded=False)
    for axis, name in enumerate(("Move X", "Move Y", "Move Z")):
      w = ParamSlider(name, -200.0, 200.0, 0.0,
                      lambda val, a=axis: self._on_translation(a, val),
                      tooltip="<b>%s</b><br>Root offset along Maya's world %s "
                      "axis, in cm." % (name, "XYZ"[axis]),
                      decimals=1, label_width=60,
                      on_release=self._push_history)
      self._trans_sliders.append(w)
      root.content_layout().addWidget(w)
    right.addWidget(root)
    right.addStretch(1)
    rhost = QtWidgets.QWidget()
    rhost.setLayout(right)
    split.addWidget(rhost, 3)
    v.addLayout(split, 1)
    return container

  # --- Animate tab -------------------------------------------------------------

  def _animate_tab(self):
    host = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(host)
    bake = CollapsibleFrame("Bake Rig", expanded=True)
    lay = bake.content_layout()
    note = QtWidgets.QLabel(
        "Turns the current body into a plain Maya rig: the 104-bone skeleton "
        "skinned with Anny's weights + a blendShape with the 52 ARKit face "
        "shapes. No Anny runtime needed; exports to FBX.")
    note.setWordWrap(True)
    lay.addWidget(note)
    row = QtWidgets.QHBoxLayout()
    self.bake_face_chk = QtWidgets.QCheckBox("52 ARKit face shapes")
    self.bake_face_chk.setChecked(True)
    self.bake_face_chk.setToolTip("<b>Face shapes</b><br>One blendShape "
                                  "target per ARKit action (eyeBlinkLeft, "
                                  "jawOpen…), keyframable, mocap-ready.")
    self.bake_pose_chk = QtWidgets.QCheckBox("Apply current pose")
    self.bake_pose_chk.setChecked(True)
    self.bake_pose_chk.setToolTip("<b>Apply pose</b><br>Pose the new joints "
                                  "like the panel (otherwise the bind pose).")
    row.addWidget(self.bake_face_chk)
    row.addWidget(self.bake_pose_chk)
    self.bake_modes_spin = QtWidgets.QSpinBox()
    self.bake_modes_spin.setRange(0, 10)
    self.bake_modes_spin.setPrefix("GNM modes/region: ")
    self.bake_modes_spin.setToolTip(
        "<b>GNM expression modes</b><br>With a GNM head, also bake the first "
        "N expression modes of each region as blendShape targets (at +2 "
        "sigma).")
    self.bake_modes_spin.setEnabled(False)
    row.addWidget(self.bake_modes_spin)
    row.addStretch(1)
    row.addWidget(self._btn("Bake Rig", "skeleton", "<b>Bake rig</b><br>"
                            "Create the rigged copy of this body.",
                            self._bake_rig))
    lay.addLayout(row)
    v.addWidget(bake)

    exp = CollapsibleFrame("Export", expanded=True)
    row = QtWidgets.QHBoxLayout()
    row.addWidget(self._btn("Export Rig (FBX)…", "download", "<b>Export FBX"
                            "</b><br>Selected baked rig (or the last one) to "
                            "FBX with skin + blendShapes.", self._export_fbx))
    row.addWidget(self._btn("Static Copy", "cube", "<b>Static copy</b><br>"
                            "Duplicate the current body as a plain mesh (no "
                            "Anny link).", self._static_copy))
    row.addStretch(1)
    exp.content_layout().addLayout(row)
    v.addWidget(exp)

    tools = CollapsibleFrame("Rig tools", expanded=False)
    row = QtWidgets.QHBoxLayout()
    row.addWidget(self._btn("Bind Pose", "restart", "<b>Bind pose</b><br>Zero "
                            "the selected rig's joint rotations (root "
                            "keeps its place).", self._rig_bind_pose))
    row.addWidget(self._btn("Panel Pose → Rig", "pose", "<b>Copy pose</b><br>"
                            "Pose the selected rig's joints like the panel "
                            "(same shape required).", self._rig_copy_pose))
    row.addStretch(1)
    tools.content_layout().addLayout(row)
    v.addWidget(tools)
    v.addStretch(1)
    return self._scroll(host)

  # --- sync ------------------------------------------------------------------

  def _sync_from_body(self):
    if not self.body:
      return
    s = self.body.state
    for k, w in self._phen_sliders.items():
      w.set_value_silent(s["phenotypes"].get(k, 0.5))
    self._sync_local()
    for k, w in self._face_sliders.items():
      w.set_value_silent(s["facial_actions"].get(k, 0.0))
    self._sync_pose()
    self._sync_head()

  def _sync_local(self):
    s = self.body.state["local_changes"]
    for k, w in self._local_sliders.items():
      w.set_value_silent(s.get(k, 0.0))

  def _sync_pose(self):
    s = self.body.state
    rot = self.body.rotation(self._bone) if self._bone else [0.0] * 3
    for a, w in enumerate(self._pose_sliders):
      w.set_value_silent(rot[a])
    for a, w in enumerate(self._trans_sliders):
      w.set_value_silent(s["translation"][a])
    for b, it in self._pose_items.items():
      f = it.font(0)
      f.setBold(b in s["rotations"])
      it.setFont(0, f)

  # --- throttled refresh -------------------------------------------------------

  def _schedule_refresh(self):
    if not self._refresh_timer.isActive():
      self._refresh_timer.start()

  def _do_refresh(self):
    try:
      if self.body:
        self.body.refresh()
    except Exception as e:
      self._show_error("Update failed", e)

  def _run(self, context, fn, sync=True, history=True, msg=None):
    """Run a discrete body action with error reporting + history."""
    if not self.body:
      return
    try:
      self._refresh_timer.stop()
      fn()
      if sync:
        self._sync_from_body()
      if history:
        self._push_history()
      if msg:
        self.status.setText(msg)
    except Exception as e:
      self._show_error(context, e)

  # --- slider callbacks ----------------------------------------------------------

  def _on_phenotype(self, key, value):
    try:
      self.body.set_phenotype(key, value, update=False)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Phenotype edit failed", e)

  def _on_local(self, label, value):
    try:
      changed = self.body.set_local(label, value,
                                    symmetry=self.sym_chk.isChecked(),
                                    update=False)
      for lab in changed:
        if lab != label and lab in self._local_sliders:
          self._local_sliders[lab].set_value_silent(value)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Local change failed", e)

  def _on_face(self, label, value):
    try:
      changed = self.body.set_face(label, value,
                                   symmetry=self.sym_chk.isChecked(),
                                   update=False)
      for lab in changed:
        if lab != label and lab in self._face_sliders:
          self._face_sliders[lab].set_value_silent(value)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Face edit failed", e)

  def _on_bone_selected(self, item, _prev=None):
    bone = item.data(0, QtCore.Qt.UserRole) if item else None
    self._bone = bone
    for w in self._pose_sliders:
      w.setEnabled(bool(bone))
    self.bone_label.setText(bone or "Select a bone")
    self._sync_pose()
    if bone and self.body:
      mirror = self.body.bone_mirror.get(bone)
      self.status.setText("Bone: %s%s" % (bone, (" (mirrors %s)" % mirror)
                                          if mirror and self.sym_chk.isChecked()
                                          else ""))

  def _on_rotation(self, axis, value):
    if not self._bone:
      return
    try:
      self.body.set_rotation(self._bone, axis, value,
                             symmetry=self.sym_chk.isChecked(), update=False)
      for b in (self._bone, self.body.bone_mirror.get(self._bone)):
        if b in self._pose_items:
          f = self._pose_items[b].font(0)
          f.setBold(b in self.body.state["rotations"])
          self._pose_items[b].setFont(0, f)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Pose edit failed", e)

  def _on_translation(self, axis, value):
    try:
      self.body.set_translation(axis, value, update=False)
      self._schedule_refresh()
    except Exception as e:
      self._show_error("Root move failed", e)

  # --- actions -------------------------------------------------------------------

  def _randomize_body(self):
    self._run("Randomize failed", lambda: self.body.randomize_phenotypes(
        seed=random.randint(0, 1 << 30), children=self.child_chk.isChecked(),
        extra=self.more_chk.isChecked()), msg="Randomized body.")

  def _reset_phenotypes(self):
    self._run("Reset failed", self.body.reset_phenotypes,
              msg="Phenotypes reset.")

  def _new_body(self):
    try:
      self._refresh_timer.stop()
      self.body = AnnyBody()
      self._hist, self._hist_i = [], -1
      self._sync_from_body()
      self._push_history()
      self.status.setText("Body: %s" % self.body.transform)
      _frame(self.body)
    except Exception as e:
      self._show_error("New body failed", e)

  def _expanded_local_labels(self):
    labs = []
    for e in self._local_frames:
      if e["frame"].isVisible() and e["frame"].is_expanded():
        labs += e["labels"]
    return labs or None

  def _randomize_local(self, labels=None):
    labels = labels or self._expanded_local_labels()
    self._run("Randomize failed", lambda: self.body.randomize_local(
        scale=self.local_amount.value(), seed=random.randint(0, 1 << 30),
        symmetric=self.sym_chk.isChecked(), labels=labels),
        msg="Randomized %d local changes." % len(labels or
                                                  self.body.local_labels))

  def _reset_local(self, labels=None):
    self._run("Reset failed", lambda: self.body.reset_local(labels),
              msg="Local changes reset.")

  def _apply_expression(self):
    name = self.expr_combo.currentText()

    def go():
      self.body.state["facial_actions"] = dict(_EXPRESSIONS[name])
      self.body.refresh()
    self._run("Expression failed", go, msg="Expression: %s." % name)

  def _randomize_face(self):
    self._run("Randomize failed", lambda: self.body.randomize_face(
        seed=random.randint(0, 1 << 30), symmetric=self.sym_chk.isChecked()),
        msg="Random expression.")

  def _reset_face(self):
    self._run("Reset failed", self.body.reset_face, msg="Face reset.")

  def _pose_preset(self, name):
    self._run("Pose preset failed", lambda: self.body.pose_preset(name),
              msg="Pose: %s." % name)

  def _reset_pose(self):
    self._run("Reset failed", self.body.reset_pose, msg="Pose reset.")

  def _reset_bone(self):
    if not self._bone:
      return

    def go():
      for b in (self._bone, self.body.bone_mirror.get(self._bone)
                if self.sym_chk.isChecked() else None):
        if b:
          self.body.state["rotations"].pop(b, None)
      self.body.refresh()
    self._run("Reset failed", go, msg="Bone %s reset." % self._bone)

  def _mirror_pose(self, side):
    self._run("Mirror failed", lambda: self.body.mirror_pose(side),
              msg="Mirrored %s side." % ("left" if side == "L" else "right"))

  def _reset_all(self):
    self._run("Reset failed", self.body.reset_all, msg="Reset to neutral.")

  def _bake_rig(self):
    from anny_maya.scene import rig
    try:
      self._refresh_timer.stop()
      self.status.setText("Baking rig…")
      QtWidgets.QApplication.processEvents()
      name = rig.bake_rig(self.body, face=self.bake_face_chk.isChecked(),
                          modes=self.bake_modes_spin.value(),
                          apply_pose=self.bake_pose_chk.isChecked())
      self._last_rig = name
      self.status.setText("Baked rig: %s" % name)
    except Exception as e:
      self._show_error("Bake Rig failed", e)

  def _target_rig(self):
    from anny_maya.scene import rig
    sel = rig.find_rigs(selected_only=True)
    if sel:
      return sel[0]
    last = getattr(self, "_last_rig", None)
    if last and mc.objExists(last):
      return last
    raise RuntimeError("Select a baked Anny rig (Animate > Bake Rig) first.")

  def _export_fbx(self):
    from anny_maya.ui import tools
    try:
      tools.export_fbx_dialog(self._target_rig())
    except Exception as e:
      self._show_error("Export failed", e)

  def _static_copy(self):
    try:
      dup = mc.duplicate(self.body.transform,
                         name=self.body.transform.split("|")[-1] + "_copy")[0]
      from anny_maya.core.body import STATE_ATTR
      if mc.attributeQuery(STATE_ATTR, node=dup, exists=True):
        mc.deleteAttr(dup + "." + STATE_ATTR)
      mc.select(dup, replace=True)
      self.status.setText("Static copy: %s" % dup)
    except Exception as e:
      self._show_error("Static copy failed", e)

  def _rig_bind_pose(self):
    from anny_maya.scene import rig
    try:
      mesh = self._target_rig()
      for j in rig.rig_joints(mesh)[1:]:
        mc.setAttr(j + ".rotate", 0, 0, 0)
      self.status.setText("Bind pose on %s." % mesh)
    except Exception as e:
      self._show_error("Bind pose failed", e)

  def _rig_copy_pose(self):
    from anny_maya.scene import rig
    try:
      mesh = self._target_rig()
      meta = self.body.worker.bake(self.body.get_state())
      rig.set_pose(rig.rig_joints(mesh), meta["posed_world"])
      self.status.setText("Panel pose copied to %s." % mesh)
    except Exception as e:
      self._show_error("Copy pose failed", e)


# --- module-level helpers ---------------------------------------------------------

def _frame(body):
  """Frame every model view on the body (no-op in batch mode)."""
  try:
    if body is None or not mc.objExists(body.transform):
      return
    mc.select(body.transform, replace=True)
    for p in (mc.getPanel(type="modelPanel") or []):
      try:
        cam = mc.modelPanel(p, query=True, camera=True)
        mc.viewFit(cam, fitFactor=0.9, animate=False)
      except Exception:
        pass
  except Exception:
    logger.debug("frame body skipped", exc_info=True)


def _open_progress(parent, text):
  """A busy dialog, force-painted so it never shows up blank/white."""
  dlg = QtWidgets.QProgressDialog(text, None, 0, 0, parent)
  dlg.setWindowTitle("Anny")
  dlg.setWindowModality(QtCore.Qt.WindowModal)
  dlg.setMinimumDuration(0)
  dlg.setMinimumWidth(320)
  dlg.setCancelButton(None)
  dlg.show()
  for _ in range(3):
    QtWidgets.QApplication.processEvents()
  dlg.repaint()
  return dlg


def _prewarm_worker():
  """Start the model worker on a thread while the dialog keeps painting."""
  import threading
  from anny_maya.core import worker as _worker
  done = {"err": None}

  def run():
    try:
      _worker.get_worker()
    except Exception as e:
      done["err"] = e

  th = threading.Thread(target=run)
  th.start()
  while th.is_alive():
    th.join(0.03)
    QtWidgets.QApplication.processEvents()
  if done["err"]:
    logger.warning("worker pre-warm failed: %s", done["err"])


def _check_tool_updates_async():
  """Courtesy, non-blocking check for a newer anny-maya; silent offline."""
  import threading
  import maya.utils
  from anny_maya.services import tool_updater as tu

  def work():
    try:
      info = tu.check()
    except Exception:
      return
    if not info.get("update_available"):
      return

    def offer():
      ans = mc.confirmDialog(
          title="anny-maya update available",
          message=("A newer anny-maya is available.\n\nInstalled: %s (%s)\n"
                   "Latest:    %s (%s)\n\nDownload now? (Later: Anny > Check "
                   "for Updates.)"
                   % (tu.short(info["installed_sha"]),
                      info["installed_date"] or "?",
                      tu.short(info["latest_sha"]), info["latest_date"])),
          button=["Download", "Skip"], defaultButton="Skip",
          cancelButton="Skip", dismissString="Skip")
      if ans != "Download":
        return
      mc.waitCursor(state=True)
      try:
        latest = tu.download_and_install(latest={
            "version": info["latest_sha"], "date": info["latest_date"],
            "zip_url": info["zip_url"]})
      except Exception as e:
        mc.waitCursor(state=False)
        mc.confirmDialog(title="anny-maya Update", icon="critical",
                         message="Update failed:\n%s" % e, button=["OK"])
        return
      mc.waitCursor(state=False)
      tu.post_update_dialog(latest)

    maya.utils.executeDeferred(offer)

  threading.Thread(target=work, daemon=True).start()


def show():
  global _WINDOW
  parent = maya_main_window()
  for w in QtWidgets.QApplication.topLevelWidgets():
    if w.objectName() == _OBJECT_NAME:
      w.close()
      w.deleteLater()
  from anny_maya.services import bootstrap
  if not bootstrap.all_available():
    if not bootstrap.ensure_all_with_dialog():
      return None
  _check_tool_updates_async()
  dlg = _open_progress(parent, "Loading Anny model…")
  try:
    _prewarm_worker()
    bodies = find_bodies(selected_only=True) or find_bodies()
    target = bodies[0] if bodies else None
    dlg.setLabelText("Building panel…")
    QtWidgets.QApplication.processEvents()
    _WINDOW = AnnyPanel(parent=parent, adopt_transform=target)
  except Exception as e:
    logger.exception("Failed to open the Anny panel")
    QtWidgets.QMessageBox.critical(
        parent, "Anny - Failed to open panel",
        "%s\n\nSee the Script Editor for the full traceback." % e)
    return None
  finally:
    dlg.close()
  _WINDOW.show()
  _frame(_WINDOW.body)
  logger.info("Opened Anny panel (%s)",
              "adopted %s" % target if target else "new body")
  return _WINDOW
