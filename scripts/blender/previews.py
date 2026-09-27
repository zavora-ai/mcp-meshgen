"""Four preview renders of a mesh (R8): front, side, back, three-quarter; transparent background, soft sky light.

args: {"glb": path, "out": {"front": path, ...}, "size": 768, "engine": "BLENDER_EEVEE", "samples": 32}
Each image is written to exactly the path given (the server passes *.partial paths and renames them).
Front looks along +Y (Blender's front view), side from +X, back along -Y, three-quarter from front-right and above.
"""
import math

import bpy
from mathutils import Vector

a = args()
fresh_scene()
meshes = import_glb(a["glb"])
lo, hi = world_bounds(meshes)
center = (lo + hi) / 2
radius = max((hi - lo).length / 2, 1e-4)

scene = bpy.context.scene
engine = set_engine(scene, a.get("engine", "BLENDER_EEVEE"))
if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
    scene.eevee.taa_render_samples = int(a.get("samples", 32))
scene.render.resolution_x = scene.render.resolution_y = int(a.get("size", 768))
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.use_file_extension = False
scene.view_settings.view_transform = "Standard"

# Soft sky: a bright uniform world gives metals something to reflect (a black world renders them black), plus a sun.
world = bpy.data.worlds.new("meshgen_sky")
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs["Color"].default_value = (0.78, 0.8, 0.84, 1.0)
bg.inputs["Strength"].default_value = 1.0
scene.world = world
sun = bpy.data.objects.new("meshgen_sun", bpy.data.lights.new("meshgen_sun", "SUN"))
sun.data.energy = 3.0
sun.data.angle = math.radians(20)
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
scene.collection.objects.link(sun)

cam_data = bpy.data.cameras.new("meshgen_cam")
cam_data.lens = 50
cam = bpy.data.objects.new("meshgen_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
distance = radius / math.sin(fov / 2) * 1.08
cam_data.clip_start = distance / 100
cam_data.clip_end = distance * 10

VIEWS = {
    "front": Vector((0, -1, 0)),
    "side": Vector((1, 0, 0)),
    "back": Vector((0, 1, 0)),
    "three_quarter": Vector((0.7, -0.7, 0.45)),
}
written = []
for view, path in a["out"].items():
    direction = VIEWS[view].normalized()
    cam.location = center + direction * distance
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    written.append(path)

result(ok=True, engine=engine, files=written, bounds_min=list(lo), bounds_max=list(hi))
