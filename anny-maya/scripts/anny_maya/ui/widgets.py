"""Reusable widgets for the Anny panel (native Maya look, no custom styling).

CollapsibleFrame: Maya-style frameLayout (bold header + disclosure arrow).
ParamSlider: one labeled horizontal slider + spin box row, like Maya's own
  attribute sliders; double-click the slider to reset it to its default.

The widgets are dumb views: they render state and forward user input through
callbacks; all decisions live in the panel (controller) / core.body (model).
"""

from __future__ import annotations

try:
  from PySide6 import QtWidgets, QtCore
except ImportError:  # Maya 2022-2024
  from PySide2 import QtWidgets, QtCore


class CollapsibleFrame(QtWidgets.QWidget):
  """Maya-style collapsible frame: a bold header with a disclosure arrow,
  content below.

    frame = CollapsibleFrame("Arms", expanded=False)
    frame.content_layout().addWidget(...)
  """

  toggled = QtCore.Signal(bool)

  def __init__(self, title, expanded=True, parent=None):
    super(CollapsibleFrame, self).__init__(parent)
    outer = QtWidgets.QVBoxLayout(self)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    self._btn = QtWidgets.QToolButton()
    self._btn.setText(title)
    self._btn.setCheckable(True)
    self._btn.setChecked(bool(expanded))
    self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    self._btn.setArrowType(QtCore.Qt.DownArrow if expanded
                           else QtCore.Qt.RightArrow)
    self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Fixed)
    self._btn.setAutoRaise(True)
    f = self._btn.font()
    f.setBold(True)
    self._btn.setFont(f)
    outer.addWidget(self._btn)

    self._body = QtWidgets.QFrame()
    self._body.setFrameShape(QtWidgets.QFrame.StyledPanel)
    self._lay = QtWidgets.QVBoxLayout(self._body)
    self._lay.setContentsMargins(8, 6, 8, 6)
    self._lay.setSpacing(2)
    self._body.setVisible(bool(expanded))
    outer.addWidget(self._body)

    self._btn.toggled.connect(self._on_toggled)

  def _on_toggled(self, on):
    self._body.setVisible(on)
    self._btn.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
    self.toggled.emit(on)

  def content_layout(self):
    return self._lay

  def set_title(self, text):
    self._btn.setText(text)

  def is_expanded(self):
    return self._btn.isChecked()

  def set_expanded(self, on):
    self._btn.setChecked(bool(on))


class _ResetSlider(QtWidgets.QSlider):
  """Horizontal slider that resets to its default on double-click."""

  def __init__(self, reset_value):
    super(_ResetSlider, self).__init__(QtCore.Qt.Horizontal)
    self._reset_value = reset_value

  def mouseDoubleClickEvent(self, event):
    self.setValue(self._reset_value)  # fires valueChanged -> callback
    event.accept()


class ParamSlider(QtWidgets.QWidget):
  """``label [=====o====] [0.50]`` - one float parameter.

  ``on_change(value)`` fires while dragging (the panel throttles mesh
  updates); ``on_release()`` fires when a drag or spin edit ends (used to
  record history steps)."""

  STEPS = 1000

  def __init__(self, label, lo, hi, default, on_change, tooltip="",
               decimals=2, label_width=150, on_release=None):
    super(ParamSlider, self).__init__()
    self.lo, self.hi, self.default = float(lo), float(hi), float(default)
    self._on_change = on_change
    self._on_release = on_release
    lay = QtWidgets.QHBoxLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    self.label = QtWidgets.QLabel(label)
    self.label.setFixedWidth(label_width)
    self.slider = _ResetSlider(self._to_int(self.default))
    self.slider.setRange(0, self.STEPS)
    self.slider.setValue(self._to_int(self.default))
    self.spin = QtWidgets.QDoubleSpinBox()
    self.spin.setDecimals(decimals)
    self.spin.setRange(self.lo, self.hi)
    self.spin.setSingleStep(10 ** -decimals * 5 if decimals >= 2 else 1.0)
    self.spin.setValue(self.default)
    self.spin.setFixedWidth(70)
    self.spin.setKeyboardTracking(False)
    lay.addWidget(self.label)
    lay.addWidget(self.slider, 1)
    lay.addWidget(self.spin)
    if tooltip:
      self.setToolTip(tooltip + "<br><i>Double-click the slider to reset.</i>")

    self.slider.valueChanged.connect(self._slider_changed)
    self.slider.sliderReleased.connect(self._released)
    self.spin.valueChanged.connect(self._spin_changed)

  def _to_int(self, v):
    t = (float(v) - self.lo) / (self.hi - self.lo)
    return int(round(max(0.0, min(1.0, t)) * self.STEPS))

  def _to_float(self, i):
    return self.lo + (self.hi - self.lo) * i / float(self.STEPS)

  def _slider_changed(self, i):
    v = self._to_float(i)
    if abs(v - self.default) < 0.5 * (self.hi - self.lo) / self.STEPS:
      v = self.default  # snap so 'reset' lands exactly on the default
    self.spin.blockSignals(True)
    self.spin.setValue(v)
    self.spin.blockSignals(False)
    self._on_change(v)
    if not self.slider.isSliderDown():
      self._released()  # click/keyboard/double-click: a discrete edit

  def _spin_changed(self, v):
    self.slider.blockSignals(True)
    self.slider.setValue(self._to_int(v))
    self.slider.blockSignals(False)
    self._on_change(float(v))
    self._released()

  def _released(self):
    if self._on_release:
      self._on_release()

  def value(self):
    return float(self.spin.value())

  def set_value_silent(self, v):
    for w, val in ((self.slider, self._to_int(v)), (self.spin, float(v))):
      w.blockSignals(True)
      w.setValue(val)
      w.blockSignals(False)

  def reset(self):
    self.set_value_silent(self.default)
