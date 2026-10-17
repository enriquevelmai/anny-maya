# Maya startup hook: add the 'Anny' menu to the main window.
#
# Placed on PYTHONPATH by ANNY.mod, so Maya auto-executes it at launch.
#
# Maya runs EVERY userSetup.py it finds in one shared namespace (__main__),
# so this file leaves nothing behind there - not even a docstring (that
# would overwrite __main__.__doc__): one uniquely named function runs and
# deletes itself, and the menu code lives in anny_maya.ui.menu. (Generic
# names like _item or build_menu were overwritten by other tools'
# userSetup.py and put the Anny menu items inside the GNM menu.)


def _anny_maya_user_setup():
  import sys
  import maya.utils

  def build():
    try:
      from anny_maya.ui import menu
      menu.build()
    except Exception as e:
      sys.stderr.write("Anny menu setup failed: %s\n" % e)

  maya.utils.executeDeferred(build)


try:
  _anny_maya_user_setup()
finally:
  del _anny_maya_user_setup
