import bpy
import mathutils
import os

# ==========================================
# CONFIGURATION
# ==========================================
BVH_FILE_PATH = r"C:\Users\lucac\Documents\Thesis_Blending\Qualitatives\Xskel_Walk_Run_1\Crab___Strafe_240.bvh"
BVH_FILE_PATH = r"C:\Users\lucac\Downloads\Alligator___Walk3_22__src_Fox_-_Walk_372__run_0.bvh"

# Set True to keep joints/bones as separate named objects (slow, but lets you
# recolor individual parts for demos). False = single joined mesh, full FPS.
SEPARATE_PARTS = False

# --- AESTHETIC COLOR PALETTE (RGBA) ---
COLORS = {
    "red":    (0.80, 0.15, 0.20, 1.0),
    "blue":   (0.15, 0.40, 0.90, 1.0),
    "green":  (0.20, 0.80, 0.40, 1.0),
    "yellow": (0.90, 0.70, 0.10, 1.0),
    "orange": (0.90, 0.40, 0.10, 1.0),
    "purple": (0.50, 0.20, 0.80, 1.0),
    "cyan":   (0.10, 0.70, 0.80, 1.0),
    "white":  (0.90, 0.90, 0.90, 1.0),
    "grey":   (0.50, 0.50, 0.50, 1.0),
    "black":  (0.05, 0.05, 0.05, 1.0)
}

JOINT_COLOR_NAME = "black"
BONE_COLOR_NAME = "orange"

# Mesh resolution — smooth shading hides polygon edges well at these values.
# Spheres: 16 segments × 8 rings  (default is 32×16, overkill for stick figures)
# Cylinders: 12 vertices           (default is 32, looks smooth when shade_smooth'd)
SPHERE_SEGMENTS = 16
SPHERE_RINGS    = 8
CYLINDER_VERTS  = 12

# Bone / Joints dimension
JOINT_RADIUS = 0.04
BONE_RADIUS = 0.02
ENDSITE_SCALE = 0.65 # scaling fac. w.r.t. "normal" joints/bones

JOINT_COLOR = COLORS.get(JOINT_COLOR_NAME.lower(), COLORS["white"])
BONE_COLOR  = COLORS.get(BONE_COLOR_NAME.lower(),  COLORS["grey"])

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
    return mat

def assign_rigid_weight(obj, bone_name):
    vg = obj.vertex_groups.new(name=bone_name)
    vg.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')


# ==========================================
# BVH NAMED END SITE PARSER
# ==========================================
def parse_bvh_hierarchy(bvh_path):
    """
    Parse the BVH HIERARCHY section and return:
      - joint_offsets: dict {joint_name: mathutils.Vector}  (BVH-space offset from parent)
      - named_endsites: list of (endsite_name, parent_joint_name, offset_vector)

    The Truebones BVH variant uses named End Sites:
        End Site #name: <SomeName>
        {
            OFFSET x y z
        }
    which appear inside any JOINT block and are silently discarded by Blender.
    """
    joint_offsets = {}
    named_endsites = []

    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    joint_stack = []
    brace_depth = 0
    joint_open_depth = []

    pending_joint_name = None   # joint whose OFFSET we're about to read
    pending_endsite_name = None
    in_endsite_block = False
    endsite_open_depth = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "MOTION":
            break

        if line == '{':
            brace_depth += 1
            if in_endsite_block and endsite_open_depth is None:
                endsite_open_depth = brace_depth
            i += 1
            continue

        if line == '}':
            if in_endsite_block and brace_depth == endsite_open_depth:
                in_endsite_block = False
                pending_endsite_name = None
                endsite_open_depth = None
            elif joint_open_depth and brace_depth == joint_open_depth[-1]:
                joint_open_depth.pop()
                if joint_stack:
                    joint_stack.pop()
            brace_depth -= 1
            i += 1
            continue

        if line.startswith("ROOT ") or line.startswith("JOINT "):
            joint_name = line.split(None, 1)[1]
            pending_joint_name = joint_name
            joint_stack.append(joint_name)
            joint_open_depth.append(brace_depth + 1)
            i += 1
            continue

        if line.startswith("End Site #name:"):
            pending_endsite_name = line[len("End Site #name:"):].strip()
            in_endsite_block = True
            endsite_open_depth = None
            i += 1
            continue

        if line == "End Site":
            in_endsite_block = True
            pending_endsite_name = None
            endsite_open_depth = None
            i += 1
            continue

        if line.startswith("OFFSET "):
            parts = line.split()
            offset = mathutils.Vector((float(parts[1]), float(parts[2]), float(parts[3])))

            if in_endsite_block:
                if pending_endsite_name is not None and joint_stack:
                    named_endsites.append((pending_endsite_name, joint_stack[-1], offset))
            elif pending_joint_name is not None:
                joint_offsets[pending_joint_name] = offset
                pending_joint_name = None
            i += 1
            continue

        i += 1

    return joint_offsets, named_endsites


def derive_bvh_to_local_rotation(armature, joint_offsets):
    """
    Derive the rotation matrix R such that R @ bvh_offset = armature_local_delta
    for any bone-to-bone offset vector.

    Using a single bone pair + rotation_difference() only constrains the axis of
    the offset vector, leaving the roll around it free — causing a small residual
    rotation error on End Sites whose offsets point in a different direction.

    Instead we collect two non-collinear bone-pair vectors, build an orthonormal
    frame in each space, then construct R = local_frame @ bvh_frame^T, which
    fully constrains all three axes.
    """
    candidates = []
    for bone in armature.data.bones:
        if bone.parent is None:
            continue
        bvh_offset = joint_offsets.get(bone.name)
        if bvh_offset is None or bvh_offset.length < 0.01:
            continue
        local_delta = bone.head_local - bone.parent.head_local
        if local_delta.length < 0.01:
            continue
        candidates.append((bvh_offset.normalized(), local_delta.normalized()))
        if len(candidates) == 2:
            break

    if len(candidates) == 2:
        bvh_a,   loc_a   = candidates[0]
        bvh_b,   loc_b   = candidates[1]

        # Build orthonormal frames via Gram-Schmidt
        # BVH frame
        bvh_x = bvh_a
        bvh_z = bvh_x.cross(bvh_b).normalized()
        if bvh_z.length < 1e-6:
            # collinear fallback — drop to single-pair
            pass
        else:
            bvh_y = bvh_z.cross(bvh_x).normalized()

            # Local frame (same structure)
            loc_x = loc_a
            loc_z = loc_x.cross(loc_b).normalized()
            loc_y = loc_z.cross(loc_x).normalized()

            # R maps BVH basis → local basis:  R = [loc_x|loc_y|loc_z] @ [bvh_x|bvh_y|bvh_z]^T
            bvh_mat = mathutils.Matrix((bvh_x, bvh_y, bvh_z)).transposed()   # columns = bvh basis
            loc_mat = mathutils.Matrix((loc_x, loc_y, loc_z)).transposed()   # columns = local basis
            R = loc_mat @ bvh_mat.inverted()
            return R.to_3x3()

    if len(candidates) >= 1:
        # Single-pair fallback
        bvh_n, local_n = candidates[0]
        return bvh_n.rotation_difference(local_n).to_matrix().to_3x3()

    print("WARNING: could not derive BVH→local rotation from bone pairs, falling back to matrix_world")
    return armature.matrix_world.to_3x3().normalized()


def endsite_world_pos(offset_bvh, anchor_bone, armature, R_bvh_to_local):
    """
    Convert a BVH-space End Site OFFSET (relative to anchor_bone's head)
    into Blender world-space position.

      armature_local = anchor_bone.head_local + R_bvh_to_local @ offset_bvh
      world          = matrix_world @ armature_local
    """
    local_pos = anchor_bone.head_local + R_bvh_to_local @ offset_bvh
    return armature.matrix_world @ local_pos


# ==========================================
# MAIN MESH BUILDER
# ==========================================
def create_mesh_skeleton(bvh_path):
    if not os.path.exists(bvh_path):
        print(f"Error: BVH file not found at {bvh_path}")
        return

    # 1. Parse BVH hierarchy before importing (no Blender state needed)
    joint_offsets, named_endsites = parse_bvh_hierarchy(bvh_path)
    print(f"Found {len(named_endsites)} named End Sites:")
    for es_name, parent, offset in named_endsites:
        print(f"  {es_name}  (parent: {parent}, offset: {offset})")

    # 2. Import BVH
    bpy.ops.import_anim.bvh(filepath=bvh_path, axis_forward='-Z', axis_up='Y')
    armature = bpy.context.active_object

    if not armature or armature.type != 'ARMATURE':
        print("Error: Imported object is not an Armature.")
        return

    bpy.ops.object.mode_set(mode='OBJECT')

    # 3. REST pose so bone heads are in bind position
    armature.data.pose_position = 'REST'
    bpy.context.view_layer.update()

    # 4. Materials
    mat_joint = create_material("Mat_Joint", JOINT_COLOR)
    mat_bone  = create_material("Mat_Bone",  BONE_COLOR)

    mesh_parts = []

    # 5. Build bone name → bone lookup
    bone_map = {bone.name: bone for bone in armature.data.bones}

    def bind_part(obj, bone_name):
        """Assign vertex weights; in SEPARATE_PARTS mode also add Armature modifier."""
        assign_rigid_weight(obj, bone_name)
        if SEPARATE_PARTS:
            mod = obj.modifiers.new(type='ARMATURE', name="Armature")
            mod.object = armature

    # 6. Generate stick figure from armature bones
    for bone in armature.data.bones:
        head_world = armature.matrix_world @ bone.head_local

        # Joint sphere — named after the bone
        bpy.ops.mesh.primitive_uv_sphere_add(radius=JOINT_RADIUS, segments=SPHERE_SEGMENTS, ring_count=SPHERE_RINGS, location=head_world)
        bpy.ops.object.shade_smooth()
        sphere = bpy.context.active_object
        sphere.name = f"Joint_{bone.name}"
        sphere.data.materials.append(mat_joint)
        bind_part(sphere, bone.name)
        mesh_parts.append(sphere)

        # Cylinder to each child
        for child in bone.children:
            child_head_world = armature.matrix_world @ child.head_local
            vec = child_head_world - head_world
            length = vec.length
            if length > 0.001:
                midpoint = (head_world + child_head_world) / 2.0
                bpy.ops.mesh.primitive_cylinder_add(radius=BONE_RADIUS, depth=length, vertices=CYLINDER_VERTS, location=midpoint)
                bpy.ops.object.shade_smooth()
                cyl = bpy.context.active_object
                cyl.name = f"Bone_{bone.name}_to_{child.name}"
                cyl.data.materials.append(mat_bone)
                cyl.rotation_euler = vec.to_track_quat('Z', 'Y').to_euler()
                bpy.context.view_layer.update()
                bind_part(cyl, bone.name)
                mesh_parts.append(cyl)

        # Leaf bone: stub cylinder + tail sphere
        if not bone.children:
            tail_world = armature.matrix_world @ bone.tail_local
            vec = tail_world - head_world
            length = vec.length
            if length > 0.001:
                midpoint = (head_world + tail_world) / 2.0
                bpy.ops.mesh.primitive_cylinder_add(radius=BONE_RADIUS, depth=length, vertices=CYLINDER_VERTS, location=midpoint)
                bpy.ops.object.shade_smooth()
                cyl = bpy.context.active_object
                cyl.name = f"Bone_{bone.name}_stub"
                cyl.data.materials.append(mat_bone)
                cyl.rotation_euler = vec.to_track_quat('Z', 'Y').to_euler()
                bpy.context.view_layer.update()
                bind_part(cyl, bone.name)
                mesh_parts.append(cyl)

                bpy.ops.mesh.primitive_uv_sphere_add(radius=JOINT_RADIUS, segments=SPHERE_SEGMENTS, ring_count=SPHERE_RINGS, location=tail_world)
                bpy.ops.object.shade_smooth()
                tail_sphere = bpy.context.active_object
                tail_sphere.name = f"Joint_{bone.name}_tail"
                tail_sphere.data.materials.append(mat_joint)
                bind_part(tail_sphere, bone.name)
                mesh_parts.append(tail_sphere)

    # 7. Derive the BVH→armature-local rotation from a known bone pair
    R_bvh_to_local = derive_bvh_to_local_rotation(armature, joint_offsets)

    # 8. Place named End Sites
    for es_name, parent_joint_name, offset_bvh in named_endsites:
        parent_bone = bone_map.get(parent_joint_name)
        if parent_bone is None:
            print(f"  WARNING: parent bone '{parent_joint_name}' not found for End Site '{es_name}'")
            continue

        world_pos    = endsite_world_pos(offset_bvh, parent_bone, armature, R_bvh_to_local)
        anchor_world = armature.matrix_world @ parent_bone.head_local

        bpy.ops.mesh.primitive_uv_sphere_add(radius=JOINT_RADIUS * ENDSITE_SCALE, segments=SPHERE_SEGMENTS, ring_count=SPHERE_RINGS, location=world_pos)
        bpy.ops.object.shade_smooth()
        es_sphere = bpy.context.active_object
        es_sphere.name = f"EndSite_{es_name}"
        es_sphere.data.materials.append(mat_joint)
        bind_part(es_sphere, parent_bone.name)
        mesh_parts.append(es_sphere)

        vec = world_pos - anchor_world
        length = vec.length
        if length > 0.001:
            midpoint = (anchor_world + world_pos) / 2.0
            bpy.ops.mesh.primitive_cylinder_add(
                radius=BONE_RADIUS * ENDSITE_SCALE, depth=length, vertices=CYLINDER_VERTS, location=midpoint)
            bpy.ops.object.shade_smooth()
            cyl = bpy.context.active_object
            cyl.name = f"EndSiteBone_{es_name}"
            cyl.data.materials.append(mat_bone)
            cyl.rotation_euler = vec.to_track_quat('Z', 'Y').to_euler()
            bpy.context.view_layer.update()
            bind_part(cyl, parent_bone.name)
            mesh_parts.append(cyl)

    if SEPARATE_PARTS:
        # 9a. Keep parts separate — group under an empty for a tidy outliner.
        #     One draw call per object: flexible for demos, slower viewport.
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        group_empty = bpy.context.active_object
        group_empty.name = f"{armature.name}_MeshBody"
        group_empty.parent = armature

        for obj in mesh_parts:
            obj.parent = group_empty
            obj.parent_type = 'OBJECT'

        print(f"Done (separate): {len(mesh_parts)} individually accessible objects.")
    else:
        # 9b. Join into a single mesh — one draw call, full FPS.
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_parts:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_parts[0]
        bpy.ops.object.join()

        combined = bpy.context.active_object
        combined.name = f"{armature.name}_MeshBody"
        combined.parent = armature
        mod = combined.modifiers.new(type='ARMATURE', name='Armature')
        mod.object = armature

        print(f"Done (joined): single mesh, {len(mesh_parts)} parts merged.")

    # 10. Switch back to POSE mode
    armature.data.pose_position = 'POSE'
    bpy.context.view_layer.update()

    armature.hide_render = True
    armature.display_type = 'WIRE'


if __name__ == "__main__":
    create_mesh_skeleton(BVH_FILE_PATH)
