"""User-configurable paths, persisted across Maya sessions via optionVar.

Keys: annyPresetsDir, annyExportsDir. Getters return an existing default
when unset so callers never deal with empty values.
"""

from __future__ import annotations

import logging
import os

from maya import cmds as mc

logger = logging.getLogger(__name__)

_PRESETS_KEY = "annyPresetsDir"
_EXPORTS_KEY = "annyExportsDir"


def _get(key, default):
  if mc.optionVar(exists=key):
    val = mc.optionVar(query=key)
    if val and os.path.isdir(val):
      return val
  return default


def _set(key, value):
  mc.optionVar(stringValue=(key, value))
  logger.info("%s = %s", key, value)


def presets_dir():
  return _get(_PRESETS_KEY, os.path.expanduser("~/Documents/maya/anny_presets"))


def set_presets_dir(path):
  _set(_PRESETS_KEY, path)


def exports_dir():
  return _get(_EXPORTS_KEY, os.path.expanduser("~/Documents/maya/anny_exports"))


def set_exports_dir(path):
  _set(_EXPORTS_KEY, path)
