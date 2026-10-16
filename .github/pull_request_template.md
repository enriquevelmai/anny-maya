## What does this PR do?

<!-- Short description of the change and why it's needed. -->

## Checklist

- [ ] Headless smoke test passes (`from anny_maya import smoke_test; smoke_test.run_headless()` under mayapy)
- [ ] GUI smoke test passes in Maya (`anny-maya/tests/gui_smoke_test.py`) if the change touches the UI
- [ ] README / tooltips updated for any user-facing change
- [ ] No Anny data, runtimes or other binary blobs committed (the repo ships code + docs only)
- [ ] Numpy port still matches official Anny (`tests/validate_vs_official.py`) if `external/_anny_core.py` changed
