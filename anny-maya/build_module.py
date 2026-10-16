#!/usr/bin/env python
"""Dev helper: run the first-run setup without Maya's dialogs.

    py -3.11 anny-maya/build_module.py          (Windows)
    python3 anny-maya/build_module.py           (Linux/macOS)

Sets up runtime/ (portable CPython or venv + numpy), downloads the pinned
naver/anny assets into external/anny_data/ and compiles the model (and with
--gnm the Google GNM head into external/gnm_data/) - exactly
what the Anny panel offers on first open. Pass --force-data to re-download.
"""

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from anny_maya.services import bootstrap  # noqa: E402


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("--force-data", action="store_true")
  ap.add_argument("--gnm", action="store_true",
                  help="also download the optional Google GNM head model")
  a = ap.parse_args()
  say = lambda m, **kw: print(m, flush=True) if kw.get("tick", True) else None
  bootstrap.ensure_runtime(say)
  bootstrap.ensure_data(say, force=a.force_data)
  bootstrap.ensure_compiled(say)
  if a.gnm:
    bootstrap.ensure_gnm(say)
  print("ready: runtime=%s data=%s compiled=%s" % (
      bootstrap.runtime_available(), bootstrap.data_available(),
      bootstrap.compiled_available()), "gnm=%s" % bootstrap.gnm_available())


if __name__ == "__main__":
  main()
