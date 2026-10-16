"""Collect and display all licenses: this module's own plus every downloaded
component (NAVER Anny code + CC0 MakeHuman/MPFB2 assets, the CPython runtime
and its pip dependencies).

``collect()`` is pure (no Qt) and testable headlessly; ``show()`` opens a
Qt viewer.
"""

from __future__ import annotations

import glob
import logging
import os

from anny_maya.core import config

logger = logging.getLogger(__name__)

_ROOT = config.MODULE_ROOT
_PARENT = os.path.dirname(_ROOT)  # repo root (holds the .md docs)


def _read(path):
  try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
      return f.read()
  except Exception:
    return None


def _read_first(name):
  for base in (_ROOT, _PARENT):
    text = _read(os.path.join(base, name))
    if text:
      return text
  return None


def _site_dirs():
  return (glob.glob(os.path.join(_ROOT, "runtime", "Lib", "site-packages"))
          + glob.glob(os.path.join(_ROOT, "runtime", "lib", "python3*",
                                   "site-packages")))


def _dep_licenses():
  out = []
  for site in _site_dirs():
    for info in sorted(glob.glob(os.path.join(site, "*.dist-info"))):
      name = os.path.basename(info).replace(".dist-info", "")
      text = None
      for pat in ("LICENSE*", "COPYING*", os.path.join("licenses", "*")):
        hits = glob.glob(os.path.join(info, pat))
        if hits:
          text = _read(hits[0])
          if text:
            break
      if not text:
        meta = _read(os.path.join(info, "METADATA")) or ""
        lines = [ln for ln in meta.splitlines()
                 if ln.startswith("License:") or "License ::" in ln]
        text = "\n".join(lines) or "(license text not found; see METADATA)"
      out.append(("Dependency: %s" % name, text))
  return out


def collect():
  """Ordered list of (title, text) covering own + all downloaded licenses."""
  items = []
  own = _read_first("LICENSE")
  if own:
    items.append(("anny-maya (Apache-2.0)", own))
  notice = _read_first("NOTICE.md")
  if notice:
    items.append(("NOTICE (attributions)", notice))
  anny = _read(os.path.join(config.DATA_ROOT, "LICENSE"))
  if anny:
    items.append(("NAVER Anny code (Apache-2.0)", anny))
  cc0 = _read(os.path.join(config.DATA_DIR, "mpfb2", "LICENSE.md"))
  if cc0:
    items.append(("MakeHuman / MPFB2 + Face Units assets (CC0)", cc0))
  rt = _read(os.path.join(_ROOT, "runtime", "LICENSE.txt"))
  if rt:
    items.append(("CPython runtime (PSF)", rt))
  items.extend(_dep_licenses())
  return items


def show(parent=None):
  try:
    from PySide6 import QtWidgets, QtGui
  except ImportError:
    from PySide2 import QtWidgets, QtGui
  if parent is None:
    from anny_maya.ui import panel as ui
    parent = ui.maya_main_window()
  items = collect()
  dlg = QtWidgets.QDialog(parent)
  dlg.setWindowTitle("Anny - Licenses")
  try:
    from anny_maya.ui import icons
    dlg.setWindowIcon(icons.window_icon())
  except Exception:
    pass
  dlg.resize(780, 560)
  lay = QtWidgets.QHBoxLayout(dlg)
  listw = QtWidgets.QListWidget()
  listw.setMaximumWidth(290)
  text = QtWidgets.QPlainTextEdit()
  text.setReadOnly(True)
  text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
  try:
    text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
  except Exception:
    pass
  for title, _ in items:
    listw.addItem(title)
  listw.currentRowChanged.connect(
      lambda row: 0 <= row < len(items) and text.setPlainText(items[row][1]))
  lay.addWidget(listw)
  lay.addWidget(text, 1)
  if items:
    listw.setCurrentRow(0)
  dlg.show()
  return dlg
