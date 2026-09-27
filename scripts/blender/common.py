"""Shared helpers for mcp-meshgen's headless Blender scripts (embedded in the server; prepended to each script)."""
import json
import sys

import bpy
from mathutils import Vector

RESULT_MARKER = "MESHGEN_RESULT "


def args():
    """The JSON object passed after `--` on the command line."""
    argv = sys.argv
    return json.loads(argv[argv.index("--") + 1]) if "--" in argv else {}


def result(**fields):
    print(RESULT_MARKER + json.dumps(fields), flush=True)


def fail(message):
    result(ok=False, error=message)
    sys.exit(1)


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        fail(f"no mesh objects in {path}")
    return meshes


def world_bounds(objects):
    """Axis-aligned bounds of the evaluated objects in world space (Blender axes, Z up)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for obj in objects:
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        mw = ev.matrix_world
        for v in mesh.vertices:
            p = mw @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi


def set_engine(scene, preferred):
    """EEVEE's id changed across versions (BLENDER_EEVEE / BLENDER_EEVEE_NEXT); fall back to Workbench."""
    ids = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    for engine in [preferred, "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"]:
        if engine in ids:
            scene.render.engine = engine
            return engine
    return scene.render.engine
