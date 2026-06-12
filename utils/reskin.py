"""
Blender Python script — run in the Scripting editor.

Problem
-------
A mesh was skinned to a "full" skeleton that includes end-site joints
(e.g. Bip01_TailNub, Bip01_SpineR, …). The pipeline produces a "cropped"
skeleton where those joints don't exist as bones. When you switch the
armature modifier to the cropped skeleton, those vertex groups become
orphaned and their vertices freeze in place.

Fix
---
For every vertex group in the mesh that has no corresponding bone in the
NEW (target) armature:
  1. Look up the bone's parent in the OLD armature.
  2. Add that vertex group's weights on top of the parent bone's vertex group.
  3. Remove the orphaned vertex group.
Then switch the armature modifier to the new skeleton.

Configuration
-------------
Set the three names below, then Run Script (Alt+P).
"""

import bpy
import mathutils  # Blender built-in — IDE may warn but it resolves fine inside Blender

# ── Edit these ─────────────────────────────────────────────────────────────
MESH_NAME        = "U3DMesh.003"         # mesh object that has the armature modifier
NEW_ARMATURE_NAME = "Cat_rep_0_#0" # armature imported from the cropped BVH
# ───────────────────────────────────────────────────────────────────────────


def get_armature_modifier(mesh_obj):
    for mod in mesh_obj.modifiers:
        if mod.type == 'ARMATURE':
            return mod
    return None


def merge_weights(mesh_obj, src_vg, dst_vg):
    """Add all weights from src_vg into dst_vg (clamped to 1.0), then remove src_vg."""
    for v in mesh_obj.data.vertices:
        try:
            w = src_vg.weight(v.index)
        except RuntimeError:
            continue  # vertex not in src group
        if w <= 0.0:
            continue
        try:
            existing = dst_vg.weight(v.index)
        except RuntimeError:
            existing = 0.0
        dst_vg.add([v.index], min(existing + w, 1.0), 'REPLACE')
    mesh_obj.vertex_groups.remove(src_vg)


def duplicate_mesh(mesh_obj):
    """
    Return a full independent copy of mesh_obj (new mesh data, same vertex groups).
    The copy is placed in the same collections as the original and named with a
    '_retargeted' suffix.
    """
    new_obj = mesh_obj.copy()
    new_obj.data = mesh_obj.data.copy()  # make mesh data single-user
    new_obj.name = mesh_obj.name + "_retargeted"
    new_obj.data.name = mesh_obj.data.name + "_retargeted"

    # Add to the same collections as the original
    for col in mesh_obj.users_collection:
        col.objects.link(new_obj)

    return new_obj


def world_bone_length(armature_obj, bone):
    """Return the world-space length of a bone, accounting for the armature's object scale."""
    mat = armature_obj.matrix_world
    head = mat @ bone.head_local
    tail = mat @ bone.tail_local
    return (tail - head).length


def compute_scale_factor(old_armature, new_armature):
    """
    Compute a uniform scale factor from old→new armature by taking the median
    ratio of matching bone world-space lengths.  Using world-space lengths
    accounts for non-unit object scales (common with BVH imports).  Using the
    median avoids outliers from very short or near-zero bones.
    """
    ratios = []
    for bone in new_armature.data.bones:
        old_bone = old_armature.data.bones.get(bone.name)
        if old_bone is None:
            continue
        old_len = world_bone_length(old_armature, old_bone)
        new_len = world_bone_length(new_armature, bone)
        if old_len > 1e-6 and new_len > 1e-6:
            ratios.append(new_len / old_len)

    if not ratios:
        print("  WARNING: no common bones found for scale estimation — scale unchanged.")
        return 1.0

    ratios.sort()
    factor = ratios[len(ratios) // 2]   # median
    print(f"  Scale factor (median of {len(ratios)} bone ratios): {factor:.6f}")
    return factor


def reparent_to_armature(mesh_obj, old_armature, new_armature):
    """
    Re-parent the mesh copy to the new armature.

    The mesh's local transform relative to its old parent armature is preserved,
    with only a uniform scale correction applied to account for size differences
    between the two skeletons.  The parent inverse is zeroed so the mesh's local
    transform directly expresses its offset from the new armature.
    """
    scale_factor = compute_scale_factor(old_armature, new_armature)

    # Capture the mesh's local transform relative to the old armature before
    # clearing the parent.  matrix_parent_inverse undoes any "keep transform"
    # offset baked in at parenting time, so we must account for it:
    #   local = parent_inverse @ matrix_local
    local_mat = mesh_obj.matrix_parent_inverse @ mesh_obj.matrix_basis

    # Apply uniform scale correction
    _, local_rot, local_scale = local_mat.decompose()
    local_loc = local_mat.to_translation()
    corrected_mat = mathutils.Matrix.LocRotScale(
        local_loc * scale_factor,
        local_rot,
        local_scale * scale_factor,
    )

    # Re-parent to new armature with zeroed parent inverse, using the
    # corrected local transform directly
    mesh_obj.parent = new_armature
    mesh_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    mesh_obj.matrix_basis = corrected_mat


def main():
    mesh_obj = bpy.data.objects.get(MESH_NAME)
    if mesh_obj is None:
        raise ValueError(f"Object '{MESH_NAME}' not found.")
    if mesh_obj.type != 'MESH':
        raise ValueError(f"'{MESH_NAME}' is not a mesh object.")

    mod = get_armature_modifier(mesh_obj)
    if mod is None:
        raise ValueError(f"'{MESH_NAME}' has no Armature modifier.")

    old_armature = mod.object
    if old_armature is None:
        raise ValueError("Armature modifier has no linked armature.")

    new_armature = bpy.data.objects.get(NEW_ARMATURE_NAME)
    if new_armature is None:
        raise ValueError(f"New armature '{NEW_ARMATURE_NAME}' not found.")
    if new_armature.type != 'ARMATURE':
        raise ValueError(f"'{NEW_ARMATURE_NAME}' is not an armature object.")

    # --- Create an independent copy of the mesh ---
    copy_obj = duplicate_mesh(mesh_obj)
    bpy.context.view_layer.update()  # ensure matrix_world is fully evaluated on the new copy
    print(f"Created mesh copy: '{copy_obj.name}'")

    # Re-parent to new skeleton with scale correction
    reparent_to_armature(copy_obj, old_armature, new_armature)
    print(f"Re-parented '{copy_obj.name}' to '{NEW_ARMATURE_NAME}'.")

    new_bone_names = {b.name for b in new_armature.data.bones}

    # Collect orphaned vertex groups on the COPY (exist in mesh but not in new armature)
    orphaned = [
        vg for vg in copy_obj.vertex_groups
        if vg.name not in new_bone_names
    ]

    if not orphaned:
        print("No orphaned vertex groups found — switching armature on copy.")
    else:
        print(f"Found {len(orphaned)} orphaned vertex group(s):")

    transferred = []
    skipped = []

    for vg in orphaned:
        vg_name = vg.name  # capture immediately — vg ref becomes invalid after removal
        bone = old_armature.data.bones.get(vg_name)
        if bone is None:
            skipped.append(vg_name)
            continue

        # Walk up the hierarchy until we find a bone that exists in the new armature
        parent = bone.parent
        while parent is not None and parent.name not in new_bone_names:
            parent = parent.parent

        if parent is None:
            skipped.append(vg_name)
            print(f"  SKIP  {vg_name!r} — no ancestor found in new armature")
            continue

        dst_vg = copy_obj.vertex_groups.get(parent.name)
        if dst_vg is None:
            dst_vg = copy_obj.vertex_groups.new(name=parent.name)

        merge_weights(copy_obj, vg, dst_vg)
        transferred.append((vg_name, parent.name))
        print(f"  OK    {vg_name!r} → {parent.name!r}")

    # Point the copy's armature modifier to the new skeleton
    copy_mod = get_armature_modifier(copy_obj)
    copy_mod.object = new_armature
    print(f"\nArmature modifier on '{copy_obj.name}' switched to '{NEW_ARMATURE_NAME}'.")
    print(f"Transferred: {len(transferred)}  |  Skipped: {len(skipped)}")
    if skipped:
        print(f"Skipped groups (no ancestor in new armature): {skipped}")


main()
