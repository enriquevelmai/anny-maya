"""Manage the long-lived Anny server subprocess and talk to it.

One resident worker (singleton) keeps the model loaded so slider drags
resolve in ~20 ms. Communication is one JSON line per request over
stdin/stdout; vertices come back as a little-endian float32 file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading

from anny_maya.core import config
from anny_maya.core import meshio

_WORKER = None
_LOCK = threading.Lock()


class AnnyWorker(object):

  def __init__(self):
    config.check_install()
    self.session_dir = tempfile.mkdtemp(prefix="anny_session_")
    self._seq = 0
    # The server's stderr goes to a file: an undrained PIPE would eventually
    # fill up and block the server mid-request.
    self.log_path = os.path.join(self.session_dir, "server.log")
    self._log = open(self.log_path, "w")
    creationflags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                     if os.name == "nt" else 0)
    self.proc = subprocess.Popen(
        [config.runtime_python(), config.SERVER_SCRIPT,
         "--session", self.session_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=self._log,
        universal_newlines=True,
        bufsize=1,
        creationflags=creationflags,
    )
    line = self.proc.stdout.readline().strip()
    if not line.startswith("READY"):
      raise RuntimeError("Anny server failed to start:\n%s\n%s"
                         % (line, self.server_log()))
    self.topology = meshio.read_topology(self.session_dir)

  def server_log(self, tail=1500):
    try:
      self._log.flush()
      with open(self.log_path) as f:
        return f.read()[-tail:]
    except Exception:
      return ""

  def alive(self):
    return self.proc.poll() is None

  def _request(self, req, expect):
    if not self.alive():
      raise RuntimeError("Anny server has exited.\n%s" % self.server_log())
    self.proc.stdin.write(json.dumps(req) + "\n")
    self.proc.stdin.flush()
    resp = self.proc.stdout.readline().strip()
    if not resp.startswith(expect):
      if not resp:
        raise RuntimeError("Anny server died:\n%s" % self.server_log())
      raise RuntimeError(resp[4:] if resp.startswith("ERR ") else resp)
    return resp[len(expect):].strip()

  @staticmethod
  def _state_req(state):
    req = {}
    for k in ("phenotypes", "local_changes", "facial_actions", "rotations",
              "translation"):
      v = state.get(k)
      if v:
        req[k] = v
    return req

  def eval(self, state):
    """Evaluate a body state dict; returns a flat array('f') of Maya verts."""
    self._seq += 1
    # Round-robin two files so a pending read never races the next write.
    out = os.path.join(self.session_dir, "verts_%d.bin" % (self._seq % 2))
    req = self._state_req(state)
    req["out"] = out
    self._request(req, "OK")
    return meshio.read_vertices(out)

  def bake(self, state):
    """Write rig data (bind mesh, face targets, weights, joints) into the
    session dir; returns the parsed rig.json."""
    req = self._state_req(state)
    req["cmd"] = "bake"
    self._request(req, "BAKED")
    with open(os.path.join(self.session_dir, "rig.json")) as f:
      return json.load(f)

  def pose_preset(self, name, state):
    req = self._state_req(state)
    req.update(cmd="pose_preset", name=str(name))
    return json.loads(self._request(req, "POSE"))

  def shutdown(self):
    try:
      if self.alive():
        self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=5)
    except Exception:
      try:
        self.proc.kill()
      except Exception:
        pass
    try:
      self._log.close()
    except Exception:
      pass


def get_worker():
  """Return the shared worker, (re)starting it if needed."""
  global _WORKER
  with _LOCK:
    if _WORKER is None or not _WORKER.alive():
      _WORKER = AnnyWorker()
    return _WORKER


def shutdown_worker():
  global _WORKER
  with _LOCK:
    if _WORKER is not None:
      _WORKER.shutdown()
      _WORKER = None
