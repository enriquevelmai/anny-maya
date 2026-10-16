"""Check GitHub for updates to THIS tool (anny-maya) and self-update in place.

The installed version is recorded in ``TOOL_VERSION.json`` at the module
root; update = download the repo zip from GitHub (by exact tag or commit
SHA - never a branch zip, which GitHub serves from a cache that can lag
behind the push) and sync its ``anny-maya/`` subfolder onto the live install.

Two update channels (chosen in the update dialog, saved in TOOL_VERSION.json):

* ``release`` (default) - follows tagged GitHub Releases (e.g. v1.0.0).
* ``dev`` - follows the tip of the ``main`` branch by commit id.

``runtime/`` and ``external/anny_data/`` are NEVER touched (they are not in
the repo zip). If an update changes the pinned Anny commit
(external/ANNY_DATA.json), the next panel open re-downloads the data.

This tool's code runs inside Maya's Python and is cached in ``sys.modules``,
so after syncing the files the updater calls ``anny_maya.reload_package()``:
the next menu action / panel open loads the new code - no restart needed.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

from anny_maya.core import config

logger = logging.getLogger(__name__)

REPO = "enriquevelmai/anny-maya"
BRANCH = "main"
MODULE_SUBDIR = "anny-maya"
_UA = {"User-Agent": "anny-maya-tool-updater"}

CHANNEL_RELEASE = "release"
CHANNEL_DEV = "dev"

_VERSION_FILE = os.path.join(config.MODULE_ROOT, "TOOL_VERSION.json")

# Entirely tool-owned subfolders: replaced wholesale (handles renamed/deleted
# files, unlike a plain overlay). Icons live inside scripts/.
_REPLACE_DIRS = ("scripts", "tests")
# external/ mixes tool code with the downloaded anny_data/: only loose
# *.py / *.json files are synced.
_EXTERNAL_PRESERVE = ("anny_data", "__pycache__")
# Describe THIS install; the repo copy must never overwrite them.
_STATE_FILES = ("TOOL_VERSION.json",)


def short(version):
  v = version or "unknown"
  return v if len(v) <= 12 else v[:8]


def installed_info():
  try:
    with open(_VERSION_FILE) as f:
      return json.load(f)
  except Exception:
    return {}


def installed_label():
  info = installed_info()
  v = info.get("tag") or info.get("sha")
  return "%s (%s)" % (short(v), info.get("channel", CHANNEL_RELEASE))


def get_channel():
  ch = installed_info().get("channel")
  return ch if ch in (CHANNEL_RELEASE, CHANNEL_DEV) else CHANNEL_RELEASE


def set_channel(channel):
  if channel not in (CHANNEL_RELEASE, CHANNEL_DEV):
    raise ValueError("unknown channel: %r" % channel)
  info = installed_info()
  info.setdefault("repo", REPO)
  info["channel"] = channel
  with open(_VERSION_FILE, "w") as f:
    json.dump(info, f, indent=2)


def latest_commit(timeout=15):
  url = "https://api.github.com/repos/%s/commits/%s" % (REPO, BRANCH)
  req = urllib.request.Request(url, headers=_UA)
  with urllib.request.urlopen(req, timeout=timeout) as r:
    data = json.load(r)
  return {"version": data["sha"],
          "date": data["commit"]["committer"]["date"],
          "zip_url": "https://github.com/%s/archive/%s.zip"
                     % (REPO, data["sha"])}


def latest_release(timeout=15):
  url = "https://api.github.com/repos/%s/releases/latest" % REPO
  req = urllib.request.Request(url, headers=_UA)
  with urllib.request.urlopen(req, timeout=timeout) as r:
    data = json.load(r)
  tag = data["tag_name"]
  return {"version": tag,
          "date": (data.get("published_at") or "")[:10] or None,
          "zip_url": "https://github.com/%s/archive/refs/tags/%s.zip"
                     % (REPO, tag)}


def check():
  """Installed vs latest for the current channel (+ update_available)."""
  channel = get_channel()
  inst = installed_info()
  installed = inst.get("tag") if channel == CHANNEL_RELEASE else inst.get("sha")
  latest = latest_release() if channel == CHANNEL_RELEASE else latest_commit()
  return {
      "channel": channel,
      "installed_sha": installed,
      "installed_date": inst.get("date"),
      "latest_sha": latest["version"],
      "latest_date": latest["date"],
      "zip_url": latest["zip_url"],
      "update_available": installed != latest["version"],
  }


def _overlay_dir(src_dir, dst_dir):
  """Copy src over dst file by file, then delete files that vanished
  upstream. Fallback when Windows refuses to rename the loaded package."""
  for root, _dirs, files in os.walk(src_dir):
    rel = os.path.relpath(root, src_dir)
    out = dst_dir if rel == "." else os.path.join(dst_dir, rel)
    os.makedirs(out, exist_ok=True)
    for fname in files:
      shutil.copy2(os.path.join(root, fname), os.path.join(out, fname))
  for root, _dirs, files in os.walk(dst_dir, topdown=False):
    if "__pycache__" in root or "_cache" in root:
      continue
    rel = os.path.relpath(root, dst_dir)
    src_root = src_dir if rel == "." else os.path.join(src_dir, rel)
    for fname in files:
      if not os.path.isfile(os.path.join(src_root, fname)):
        try:
          os.remove(os.path.join(root, fname))
        except OSError:
          pass
    if not os.path.isdir(src_root):
      try:
        os.rmdir(root)
      except OSError:
        pass


def _replace_dir(src_dir, dst_dir):
  """Wholesale replace (rename old aside, move new in); if Windows refuses
  the rename (Maya holds the loaded package), overlay in place instead."""
  if not os.path.isdir(src_dir):
    return
  if not os.path.isdir(dst_dir):
    shutil.move(src_dir, dst_dir)
    return
  old = dst_dir + "_old"
  if os.path.isdir(old):
    shutil.rmtree(old, ignore_errors=True)
  try:
    os.rename(dst_dir, old)
  except OSError as e:
    logger.info("Rename refused for %s (%s) - overlaying in place", dst_dir, e)
    _overlay_dir(src_dir, dst_dir)
    return
  try:
    shutil.move(src_dir, dst_dir)
  except Exception:
    if not os.path.isdir(dst_dir):
      os.rename(old, dst_dir)
    raise
  shutil.rmtree(old, ignore_errors=True)


def _sync_external(src_external, dst_external):
  """Sync loose *.py / *.json under external/ (add/update/remove), never
  touching the downloaded anny_data/."""
  os.makedirs(dst_external, exist_ok=True)
  keep = lambda f: (f.endswith((".py", ".json"))
                    and f not in _EXTERNAL_PRESERVE)
  src_names = set()
  if os.path.isdir(src_external):
    for fname in os.listdir(src_external):
      if keep(fname) and os.path.isfile(os.path.join(src_external, fname)):
        src_names.add(fname)
        shutil.copy2(os.path.join(src_external, fname),
                     os.path.join(dst_external, fname))
  for fname in list(os.listdir(dst_external)):
    if (keep(fname) and fname not in src_names
        and os.path.isfile(os.path.join(dst_external, fname))):
      os.remove(os.path.join(dst_external, fname))


def _sync_top_level(src_root, dst_root):
  """Overwrite loose top-level files (README, NOTICE, LICENSE...)."""
  for entry in os.listdir(src_root):
    src_path = os.path.join(src_root, entry)
    if os.path.isdir(src_path) or entry.startswith("."):
      continue
    if entry in _STATE_FILES:
      continue
    shutil.copy2(src_path, os.path.join(dst_root, entry))


def download_and_install(timeout=180, latest=None):
  """Download the channel's repo zip and sync anny-maya/ onto this install.

  ``latest`` reuses a version the caller already resolved, so the files
  installed and the version recorded always describe the same commit."""
  if not os.access(config.MODULE_ROOT, os.W_OK):
    raise RuntimeError("Module folder is not writable: %s" % config.MODULE_ROOT)
  channel = get_channel()
  if latest is None:
    latest = latest_release() if channel == CHANNEL_RELEASE else latest_commit()
  logger.info("Downloading %s ...", latest["zip_url"])
  headers = dict(_UA)
  headers["Cache-Control"] = "no-cache"
  req = urllib.request.Request(latest["zip_url"], headers=headers)
  with urllib.request.urlopen(req, timeout=timeout) as r:
    blob = r.read()

  tmp = tempfile.mkdtemp(prefix="anny_tool_update_")
  try:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
      z.extractall(tmp)
    roots = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
    if not roots:
      raise RuntimeError("Downloaded archive was empty.")
    module_src = os.path.join(tmp, roots[0], MODULE_SUBDIR)
    if not os.path.isdir(os.path.join(module_src, "scripts", "anny_maya")):
      raise RuntimeError("Downloaded archive is not an anny-maya repo.")
    for name in _REPLACE_DIRS:
      _replace_dir(os.path.join(module_src, name),
                   os.path.join(config.MODULE_ROOT, name))
    _sync_external(os.path.join(module_src, "external"), config.EXTERNAL_DIR)
    _sync_top_level(module_src, config.MODULE_ROOT)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)

  stamp = {"repo": REPO, "branch": BRANCH, "channel": channel,
           "tag": None, "sha": None, "date": latest["date"]}
  key = "tag" if channel == CHANNEL_RELEASE else "sha"
  stamp[key] = latest["version"]
  with open(_VERSION_FILE, "w") as f:
    json.dump(stamp, f, indent=2)
  if installed_info().get(key) != latest["version"]:
    raise RuntimeError(
        "Files updated but the version could not be recorded in %s, so the "
        "update prompt would keep coming back. Check the file is writable."
        % _VERSION_FILE)

  try:
    from anny_maya.core import worker
    worker.shutdown_worker()
  except Exception:
    pass
  logger.info("Updated anny-maya to %s (%s)", short(latest["version"]),
              latest["date"])
  try:
    import anny_maya
    anny_maya.reload_package()
  except Exception:
    logger.warning("package reload after update failed; restart Maya",
                   exc_info=True)
  return latest


def _channel_dialog():
  from maya import cmds as mc
  current = get_channel()
  ans = mc.confirmDialog(
      title="anny-maya update channel",
      message=("Choose which updates to follow (current: %s).\n\n"
               "Releases - stable, published versions only (recommended).\n"
               "Dev - the latest commits on main, before they are\n"
               "packaged in a release." % current),
      button=["Releases (stable)", "Dev (latest commits)", "Cancel"],
      defaultButton="Releases (stable)",
      cancelButton="Cancel", dismissString="Cancel")
  picked = {"Releases (stable)": CHANNEL_RELEASE,
            "Dev (latest commits)": CHANNEL_DEV}.get(ans)
  if picked is None or picked == current:
    return False
  set_channel(picked)
  return True


def show_update_dialog(parent=None):
  """Interactive check + optional download, via Maya confirm dialogs."""
  from maya import cmds as mc
  channel = get_channel()
  try:
    info = check()
  except Exception as e:
    logger.exception("tool update check failed")
    mc.confirmDialog(title="anny-maya Update", icon="critical",
                     message="Could not check for updates:\n%s" % e,
                     button=["OK"])
    return
  if not info["update_available"]:
    ans = mc.confirmDialog(
        title="anny-maya Update",
        message="You already have the latest anny-maya.\n\nInstalled: "
                "%s (%s)\nChannel: %s"
                % (short(info["installed_sha"]),
                   info["installed_date"] or "?", channel),
        button=["OK", "Change Channel..."], defaultButton="OK",
        cancelButton="OK", dismissString="OK")
    if ans == "Change Channel..." and _channel_dialog():
      show_update_dialog(parent)
    return
  ans = mc.confirmDialog(
      title="anny-maya Update available",
      message=("A newer anny-maya is available on the '%s' channel.\n\n"
               "Installed: %s (%s)\nLatest:    %s (%s)\n\nDownload and "
               "install now?"
               % (channel, short(info["installed_sha"]),
                  info["installed_date"] or "?",
                  short(info["latest_sha"]), info["latest_date"])),
      button=["Download", "Change Channel...", "Cancel"],
      defaultButton="Download", cancelButton="Cancel", dismissString="Cancel")
  if ans == "Change Channel...":
    if _channel_dialog():
      show_update_dialog(parent)
    return
  if ans != "Download":
    return
  mc.waitCursor(state=True)
  try:
    latest = download_and_install(latest={
        "version": info["latest_sha"], "date": info["latest_date"],
        "zip_url": info["zip_url"]})
  except Exception as e:
    logger.exception("tool update download failed")
    mc.waitCursor(state=False)
    mc.confirmDialog(title="anny-maya Update", icon="critical",
                     message="Update failed:\n%s" % e, button=["OK"])
    return
  mc.waitCursor(state=False)
  post_update_dialog(latest)


def post_update_dialog(latest):
  from maya import cmds as mc
  ans = mc.confirmDialog(
      title="anny-maya updated", icon="information",
      message=("anny-maya updated to %s (%s).\n\nThe new code has been "
               "reloaded - close and re-open the Anny panel to use it. "
               "Restarting Maya is optional."
               % (short(latest["version"]), latest["date"])),
      button=["Continue", "Restart Maya"], defaultButton="Continue",
      cancelButton="Continue", dismissString="Continue")
  if ans == "Restart Maya":
    restart_maya()


def restart_maya():
  """Best-effort: launch a fresh Maya, then quit this one."""
  from maya import cmds as mc
  loc = os.environ.get("MAYA_LOCATION")
  exe = os.path.join(loc, "bin", "maya.exe" if os.name == "nt" else "maya") \
      if loc else None
  if not exe or not os.path.isfile(exe):
    mc.confirmDialog(title="Restart Maya", icon="warning",
                     message="Couldn't locate the Maya executable.\n"
                             "Please restart Maya manually.", button=["OK"])
    return
  if mc.file(query=True, modified=True):
    ans = mc.confirmDialog(
        title="Unsaved changes",
        message="Your scene has unsaved changes that may be lost on restart.",
        button=["Restart Anyway", "Cancel"], cancelButton="Cancel",
        dismissString="Cancel")
    if ans != "Restart Anyway":
      return
  try:
    subprocess.Popen([exe])
  except Exception as e:
    mc.confirmDialog(title="Restart Maya", icon="critical",
                     message="Could not launch a new Maya:\n%s" % e,
                     button=["OK"])
    return
  mc.evalDeferred(lambda: mc.quit(force=True))
