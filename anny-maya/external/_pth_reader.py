"""Torch-free reader for PyTorch ``.pth`` zip checkpoints (numpy only).

A modern ``torch.save`` file is a zip holding ``<name>/data.pkl`` (a pickle
whose tensors are *persistent ids* pointing at raw storages) plus one raw
little-endian blob per storage under ``<name>/data/<key>``. We unpickle with a
restricted ``find_class`` that only resolves the handful of torch rebuild
helpers (mapped to numpy builders) and plain containers — anything else is
refused, so loading never imports torch nor executes arbitrary globals.

    data = load_pth("anny.pth")   # -> dict of numpy arrays / lists / str
"""

from __future__ import annotations

import collections
import pickle
import zipfile

import numpy as np

# torch storage class name -> numpy dtype
_STORAGE_DTYPES = {
    "DoubleStorage": np.float64,
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "BFloat16Storage": None,  # not representable in numpy; refused below
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType(object):
  def __init__(self, name):
    self.name = name
    self.dtype = _STORAGE_DTYPES.get(name)


def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad=False, backward_hooks=None,
                       metadata=None):
  arr = storage
  itemsize = arr.dtype.itemsize
  if len(size) == 0:
    return arr[storage_offset].copy()
  out = np.lib.stride_tricks.as_strided(
      arr[storage_offset:], shape=tuple(size),
      strides=tuple(s * itemsize for s in stride))
  return np.array(out)  # own the memory, contiguous


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
  return data


_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
}


class _Unpickler(pickle.Unpickler):

  def __init__(self, fileobj, zf, prefix):
    super(_Unpickler, self).__init__(fileobj)
    self._zf = zf
    self._prefix = prefix
    self._cache = {}

  def find_class(self, module, name):
    if (module, name) in _ALLOWED:
      return _ALLOWED[(module, name)]
    if module == "torch" and name.endswith("Storage"):
      return _StorageType(name)
    raise pickle.UnpicklingError("refusing to load global %s.%s"
                                 % (module, name))

  def persistent_load(self, pid):
    # ('storage', storage_type, key, location, numel)
    if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
      raise pickle.UnpicklingError("unsupported persistent id %r" % (pid,))
    stype, key = pid[1], pid[2]
    if key in self._cache:
      return self._cache[key]
    dtype = getattr(stype, "dtype", None)
    if dtype is None:
      raise pickle.UnpicklingError("unsupported storage %r"
                                   % getattr(stype, "name", stype))
    raw = self._zf.read("%sdata/%s" % (self._prefix, key))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
    arr = arr.astype(dtype, copy=False)
    self._cache[key] = arr
    return arr


def load_pth(path):
  """Load a ``torch.save`` zip checkpoint into numpy/py objects."""
  with zipfile.ZipFile(path) as zf:
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
    if not pkl:
      raise ValueError("%s is not a torch zip checkpoint" % path)
    name = pkl[0]
    prefix = name[: -len("data.pkl")]
    with zf.open(name) as f:
      return _Unpickler(f, zf, prefix).load()
