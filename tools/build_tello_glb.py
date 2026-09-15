#!/usr/bin/env python3
"""
build_tello_glb.py
Converts and assembles the low-poly DJI Tello model from tello_repo
into a clean, simulation-ready glTF 2.0 binary (resources/tello.glb).

Export Order & Structure:
  Mesh 0: TelloBody   (Chassis centered at (0, 0, 0), colored in visible gray) [1 primitive]
  Mesh 1: Propeller_0 (Motor 0: +X, +Y) [1 primitive, level & seated on shaft]
  Mesh 2: Propeller_1 (Motor 1: +X, -Y) [1 primitive, level & seated on shaft]
  Mesh 3: Propeller_2 (Motor 2: -X, -Y) [1 primitive, level & seated on shaft]
  Mesh 4: Propeller_3 (Motor 3: -X, +Y) [1 primitive, level & seated on shaft]
"""

import bpy
import mathutils
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLEND_SOURCE = os.path.join(ROOT_DIR, "tello_repo", "FromWeb", "Low", "LowPolyTello.blend")
OUTPUT_GLB = os.path.join(ROOT_DIR, "resources", "tello.glb")
OUTPUT_BLEND = os.path.join(ROOT_DIR, "resources", "tello.blend")

def get_or_create_material(name, color_rgba, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color_rgba
            bsdf.inputs['Roughness'].default_value = roughness
            bsdf.inputs['Metallic'].default_value = metallic
    else:
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].default_value = color_rgba
                bsdf.inputs['Roughness'].default_value = roughness
                bsdf.inputs['Metallic'].default_value = metallic
    return mat

def main():
    print(f"Loading {BLEND_SOURCE}...")
    bpy.ops.wm.open_mainfile(filepath=BLEND_SOURCE)

    body = bpy.data.objects.get("TelloBody")
    blade = bpy.data.objects.get("Blade")

    if not body or not blade:
        print("Error: Could not find TelloBody or Blade in blend file!", file=sys.stderr)
        sys.exit(1)

    # 1. Seated motor shaft positions in WORLD space:
    # Motor collar top is at Z=0.8374, shaft tip is at Z=0.9131.
    # Placing hub center at Z=0.8689 seats the hub base (hub_bottom_z = -0.0315)
    # directly on the collar (0.8689 - 0.0315 = 0.8374).
    PROPELLER_SEATING_Z = 0.8689
    motor_shaft_positions = [
        mathutils.Vector(( 1.5379,  1.0319, PROPELLER_SEATING_Z)), # Motor 0: +X, +Y
        mathutils.Vector(( 1.3409, -1.4147, PROPELLER_SEATING_Z)), # Motor 1: +X, -Y
        mathutils.Vector((-1.3127, -1.3670, PROPELLER_SEATING_Z)), # Motor 2: -X, -Y
        mathutils.Vector((-1.3397,  0.9200, PROPELLER_SEATING_Z)), # Motor 3: -X, +Y
    ]

    center_xy = sum(motor_shaft_positions, mathutils.Vector((0, 0, 0))) / 4.0
    # True world Z center of mass for the body:
    world_z = [(body.matrix_world @ v.co).z for v in body.data.vertices]
    z_mid = (min(world_z) + max(world_z)) / 2.0
    center_offset = mathutils.Vector((center_xy.x, center_xy.y, z_mid))

    # Scale: DJI Tello wheelbase ~0.098m diagonal. 1 unit in blend ~ 1 inch (0.0254m)
    SCALE = 0.0254
    print(f"Centering offset: {center_offset}, Scale factor: {SCALE}")

    # Materials
    mat_gray = get_or_create_material("TelloGray", (0.58, 0.58, 0.60, 1.0), roughness=0.5)
    mat_blade = get_or_create_material("TelloBlade", (0.22, 0.22, 0.22, 1.0), roughness=0.5)

    # 2. Transform body vertices from WORLD space -> CENTERED & SCALED space
    body_matrix = body.matrix_world.copy()
    for v in body.data.vertices:
        world_co = body_matrix @ v.co
        v.co = (world_co - center_offset) * SCALE

    body.location = (0, 0, 0)
    body.rotation_euler = (0, 0, 0)
    body.scale = (1, 1, 1)

    body.data.materials.clear()
    body.data.materials.append(mat_gray)
    for f in body.data.polygons:
        f.material_index = 0

    # 3. Prepare Propeller Prototype
    # In LowPolyTello.blend, the blade mesh has a built-in X-tilt of ~ -12.79 deg
    # that was originally counter-rotated by rotation_euler.x = 0.2232 rad.
    # We apply this rotation directly to the vertex coordinates to level the blade horizontally.
    rot_x = mathutils.Euler((blade.rotation_euler.x, 0, 0)).to_matrix()

    # Find the hub center (vertices 6 to 17 form the central cylinder)
    hub_v = [rot_x @ blade.data.vertices[i].co for i in range(6, 18)]
    hub_center = sum(hub_v, mathutils.Vector((0, 0, 0))) / len(hub_v)

    # Level, center in local XY/Z, and scale
    blade_mesh = blade.data
    blade_mesh.materials.clear()
    blade_mesh.materials.append(mat_blade)
    for f in blade_mesh.polygons:
        f.material_index = 0

    for v in blade_mesh.vertices:
        leveled_co = rot_x @ v.co - hub_center
        v.co = leveled_co * SCALE

    # Create 4 independent propellers (Propeller_0 to Propeller_3)
    prop_objects = []
    for i, m_pos in enumerate(motor_shaft_positions):
        target_pos = (m_pos - center_offset) * SCALE

        new_mesh = blade_mesh.copy()
        new_mesh.name = f"TelloPropeller_{i}"

        # Place the blade vertices seated directly on the motor shaft
        for v in new_mesh.vertices:
            v.co = v.co + target_pos

        prop_obj = bpy.data.objects.new(f"Propeller_{i}", new_mesh)
        prop_obj.location = (0, 0, 0)
        prop_obj.rotation_euler = (0, 0, 0)
        prop_obj.scale = (1, 1, 1)
        bpy.context.collection.objects.link(prop_obj)
        prop_objects.append(prop_obj)

    # Remove template blade
    bpy.data.objects.remove(blade, do_unlink=True)

    # Select in order: Body first (Mesh 0), then Propellers 0..3 (Meshes 1..4)
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    for p in prop_objects:
        p.select_set(True)
    bpy.context.view_layer.objects.active = body

    # Save clean Blender source file
    os.makedirs(os.path.dirname(OUTPUT_BLEND), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
    print(f"Saved Blender model to: {OUTPUT_BLEND}")

    # Export to GLB
    os.makedirs(os.path.dirname(OUTPUT_GLB), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_GLB,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_materials='EXPORT',
        export_yup=False
    )
    print(f"Successfully exported GLB to: {OUTPUT_GLB}")

if __name__ == "__main__":
    main()
