"""Engine-ready copy of a mesh (R9): scale to real metres, origin at the bottom centre, transforms applied, Z up.

args: {"glb": path, "out_glb": path, "out_fbx": path or null, "size_m": 1.2 or null, "size_axis": "longest"|"height"}
Writes exactly the paths given (the server passes *.partial paths). Decimation happens before the texture bake in the
worker, so this script never touches topology. The FBX uses the settings verified for Unreal 5.8's importer:
global_scale 1.0, apply_unit_scale False, FBX_SCALE_NONE, forward -Z, up Y, face smoothing, no leaf bones, textures embedded.
"""
import os
import shutil
import tempfile

import bpy
from mathutils import Matrix, Vector

a = args()
fresh_scene()
meshes = import_glb(a["glb"])

# Flatten the hierarchy: world transforms baked into the meshes, helper empties removed.
for obj in meshes:
    mw = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = mw
for obj in list(bpy.context.scene.objects):
    if obj.type != "MESH":
        bpy.data.objects.remove(obj, do_unlink=True)
for obj in meshes:
    if obj.data.users > 1:
        obj.data = obj.data.copy()  # shared data would be transformed once per user
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)

lo, hi = world_bounds(meshes)
dims = hi - lo
scale = 1.0
size_m = a.get("size_m")
if size_m:
    reference = dims.z if a.get("size_axis") == "height" else max(dims)
    if reference <= 0:
        fail("the mesh has no extent to scale")
    scale = float(size_m) / reference

# Scale about the bottom centre, then move that point to the origin.
pivot = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
xform = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-pivot)
for obj in meshes:
    obj.data.transform(xform)
    obj.data.update()
lo2, hi2 = world_bounds(meshes)

tmp = tempfile.mkdtemp(prefix="meshgen-post-")
try:
    tmp_glb = os.path.join(tmp, "game.glb")
    bpy.ops.export_scene.gltf(filepath=tmp_glb, export_format="GLB", use_selection=False, export_yup=True)
    shutil.move(tmp_glb, a["out_glb"])
    fbx = a.get("out_fbx")
    if fbx:
        tmp_fbx = os.path.join(tmp, "game.fbx")
        bpy.ops.export_scene.fbx(
            filepath=tmp_fbx, use_selection=False, global_scale=1.0, apply_unit_scale=False,
            apply_scale_options="FBX_SCALE_NONE", axis_forward="-Z", axis_up="Y", mesh_smooth_type="FACE",
            add_leaf_bones=False, path_mode="COPY", embed_textures=True,
        )
        shutil.move(tmp_fbx, fbx)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

result(ok=True, scale=scale, bounds_min=list(lo2), bounds_max=list(hi2), size=list(hi2 - lo2))
