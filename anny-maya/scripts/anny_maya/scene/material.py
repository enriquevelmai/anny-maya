"""Per-part materials for Anny bodies (skin / eyes / tongue).

Uses Maya's native **standardSurface** (2020+; Arnold renders it natively) -
NOT aiStandardSurface: Maya's vertex-colour tools neither paint nor display
through Arnold's shader. Falls back to lambert on older Mayas. Shaders are
shared by every Anny body in the scene.
"""

from __future__ import annotations

from maya import cmds as mc

# mesh part (topology.json part_names) -> material name
MATERIAL_OF_PART = {"body": "skin", "eye_l": "eyes", "eye_r": "eyes",
                    "tongue": "tongue"}

# material -> (RGB, specular, roughness, subsurface)
_PALETTE = {
    "skin": ((0.80, 0.60, 0.50), 0.25, 0.5, 0.15),
    "eyes": ((0.92, 0.92, 0.94), 0.8, 0.1, 0.0),
    "tongue": ((0.78, 0.40, 0.42), 0.5, 0.3, 0.1),
}

COMPONENT_GROUPS = {"Eyes": ("eyes",), "Tongue": ("tongue",)}


def _standard_surface_available():
  try:
    return "standardSurface" in (mc.allNodeTypes() or [])
  except Exception:
    return False


def shader_name(component):
  return "anny_%s_mat" % component


def _new_standard_surface(name):
  """standardSurface node, or None. (Maya 2022's shadingNode can return
  None for it in standalone sessions; createNode still works there.)"""
  if not _standard_surface_available():
    return None
  for make in (lambda: mc.shadingNode("standardSurface", asShader=True,
                                      name=name),
               lambda: mc.createNode("standardSurface", name=name)):
    try:
      sh = make()
    except Exception:
      sh = None
    if sh:
      return sh
  return None


def _make_shader(name, color, spec, rough, sss):
  sh = _new_standard_surface(name)
  if sh:
    mc.setAttr(sh + ".base", 1.0)
    mc.setAttr(sh + ".baseColor", *color, type="double3")
    mc.setAttr(sh + ".specular", spec)
    mc.setAttr(sh + ".specularRoughness", rough)
    if sss:
      mc.setAttr(sh + ".subsurface", sss)
      mc.setAttr(sh + ".subsurfaceColor", *color, type="double3")
  else:
    sh = mc.shadingNode("lambert", asShader=True, name=name)
    mc.setAttr(sh + ".color", *color, type="double3")
  return sh


def get_or_create(component):
  """Shading group for ``component`` (created once, then reused)."""
  color, spec, rough, sss = _PALETTE.get(component, _PALETTE["skin"])
  name = shader_name(component)
  sg_name = name + "SG"
  if mc.objExists(sg_name):
    return sg_name
  sh = _make_shader(name, color, spec, rough, sss)
  sg = mc.sets(renderable=True, noSurfaceShader=True, empty=True, name=sg_name)
  mc.connectAttr(sh + ".outColor", sg + ".surfaceShader", force=True)
  return sg


def set_component_visible(label, visible):
  """Show/hide a part of every Anny body via shader opacity (one mesh, so
  faces can't be hidden individually). Viewport 2.0 honours it."""
  for comp in COMPONENT_GROUPS.get(label, ()):
    sh = shader_name(comp)
    if not mc.objExists(sh):
      continue
    v = 1.0 if visible else 0.0
    if mc.nodeType(sh) == "standardSurface":
      mc.setAttr(sh + ".opacity", v, v, v, type="double3")
    else:
      mc.setAttr(sh + ".transparency", 1 - v, 1 - v, 1 - v, type="double3")


def component_visible(label):
  for comp in COMPONENT_GROUPS.get(label, ()):
    sh = shader_name(comp)
    if mc.objExists(sh):
      if mc.nodeType(sh) == "standardSurface":
        return mc.getAttr(sh + ".opacity")[0][0] > 0.5
      return mc.getAttr(sh + ".transparency")[0][0] < 0.5
  return True


def set_skin_color(rgb):
  sh = shader_name("skin")
  if not mc.objExists(sh):
    get_or_create("skin")
  attr = "baseColor" if mc.nodeType(sh) == "standardSurface" else "color"
  mc.setAttr("%s.%s" % (sh, attr), *rgb, type="double3")
  if mc.nodeType(sh) == "standardSurface":
    mc.setAttr(sh + ".subsurfaceColor", *rgb, type="double3")
