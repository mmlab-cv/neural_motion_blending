"""
nla_blend.py — True Blender NLA-editor baseline for motion blending.

Sets up two NLA strips on the same armature (ref on bottom, tgt on top),
animates both strips' influence via keyframe_insert (use_animated_influence=True)
to replicate blend_in/blend_out behavior with customizable easing profiles.
Bakes the NLA result with manual frame sampling and exports as BVH.

Invocation (background mode):
    blender --background --python nla_blend.py -- [args]
"""

import sys
import os
import re
import math
import json
import argparse
from pathlib import Path

import bpy

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--ref_motion",  type=str, default="")
    parser.add_argument("--tgt_motion",  type=str, default="")
    parser.add_argument("--ref_samples", type=str, default="")
    parser.add_argument("--tgt_samples", type=str, default="")
    parser.add_argument("--bvh_dir",     type=str, default="")
    parser.add_argument("--dataset_npy_dir",
                        default="./dataset/truebones/zoo/truebones_processed/motions")
    parser.add_argument("--output_dir",  default="./save/nla_blend", type=str)

    parser.add_argument("--alpha_values",       default=[1.0], type=float, nargs="+")
    parser.add_argument("--blend_schedule",     default="ease",
                        choices=["static", "linear", "ease"])
    parser.add_argument("--ease_slope",         default=1.0, type=float)
    parser.add_argument("--control_mode",       default="both",
                        choices=["ref", "tgt", "both"])

    parser.add_argument("--ref_frame_start", default=0, type=int)
    parser.add_argument("--tgt_frame_start", default=0, type=int)
    parser.add_argument("--overlap_length",  default=-1, type=int)
    parser.add_argument("--ref_num_loops",   default=1, type=int)
    parser.add_argument("--tgt_num_loops",   default=1, type=int)

    parser.add_argument("--scale",        default=0.01,  type=float)
    parser.add_argument("--seed",         default=10,    type=int)
    parser.add_argument("--overlap_only", action="store_true",
                        help="Trim output to the overlap region only (min shared length)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def import_bvh(bvh_path: str, scale: float = 0.01):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=scale, use_fps_scale=False,
                             axis_forward='Y', axis_up='Z')
    arm_obj = bpy.context.selected_objects[0]
    action  = arm_obj.animation_data.action
    return arm_obj, action


def check_skeletons_match(arm_a, arm_b) -> bool:
    return {b.name for b in arm_a.pose.bones} == {b.name for b in arm_b.pose.bones}


# ---------------------------------------------------------------------------
# Influence schedule
# ---------------------------------------------------------------------------

def _ease_fn(t: float, slope: float) -> float:
    """Symmetric ease-in-out curve, always in [0, 1].

    slope=1 → standard sine ease.  slope>1 → sharper S-curve (spends more
    time near 0/1 and crosses faster).  Implemented as a power curve applied
    symmetrically on both halves so the output is always clamped to [0, 1].
    """
    s = 0.5 * (1.0 + math.sin(math.pi * (t - 0.5)))   # standard ease, s∈[0,1]
    if slope == 1.0:
        return s
    if s <= 0.5:
        return 0.5 * (2.0 * s) ** slope
    else:
        return 1.0 - 0.5 * (2.0 * (1.0 - s)) ** slope


def build_influence_keypoints(ref_s, ref_e, tgt_s, tgt_e,
                               alpha, blend_schedule, ease_slope):
    """
    Return (ref_keys, tgt_keys) — lists of (frame, influence) keypoints.

    Mimics exactly: 
      - ref.blend_out = OVERLAP
      - tgt.blend_in  = OVERLAP
    Using F-Curves so we can apply ease/bezier slopes.
    """
    overlap_s = max(ref_s, tgt_s)
    overlap_e = min(ref_e, tgt_e)

    if blend_schedule == "static":
        ref_keys = [(ref_s, alpha), (ref_e - 1, alpha)]
        tgt_keys = [(tgt_s, alpha), (tgt_e - 1, alpha)]
        return ref_keys, tgt_keys

    def _ramp(start_f, end_f, v_start, v_end):
        if blend_schedule == "linear":
            return [(start_f, v_start), (end_f, v_end)]
        # ease: dense samples so Blender's BEZIER handles are well-conditioned
        n = end_f - start_f
        pts = []
        for i in range(n + 1):
            t = i / max(n, 1)
            v = v_start + (v_end - v_start) * _ease_fn(t, ease_slope)
            pts.append((start_f + i, v))
        return pts

    if ref_s <= tgt_s:
        # ref starts first (typical temporal transition)
        ref_keys = [(ref_s, alpha)] + _ramp(overlap_s, overlap_e, alpha, 0.0) + [(ref_e - 1, 0.0)]
        tgt_keys = [(tgt_s, 0.0)]  + _ramp(overlap_s, overlap_e, 0.0, alpha) + [(tgt_e - 1, alpha)]
    else:
        # tgt starts first (reverse transition)
        tgt_keys = [(tgt_s, alpha)] + _ramp(overlap_s, overlap_e, alpha, 0.0) + [(tgt_e - 1, 0.0)]
        ref_keys = [(ref_s, 0.0)]   + _ramp(overlap_s, overlap_e, 0.0, alpha) + [(ref_e - 1, alpha)]

    return ref_keys, tgt_keys


def _set_fcurve_interpolation(strip, interp: str):
    """Set keyframe interpolation on all F-Curves of the given NLA strip."""
    for fc in strip.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _set_bezier_auto_handles(strip):
    """Set AUTO_CLAMPED handles on all keypoints (smooth ease in/out)."""
    for fc in strip.fcurves:
        for kp in fc.keyframe_points:
            kp.handle_left_type  = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
        fc.update()


# ---------------------------------------------------------------------------
# Core blend routine
# ---------------------------------------------------------------------------

def _patch_end_sites(out_bvh_path, ref_bvh_path):
    """
    Add named End Site blocks from the GT reference BVH to the exported BVH.

    GT BVH files (Motion-library format) store named End Sites as siblings of
    regular child joints (e.g. ear bones as End Sites of the head joint).
    Blender's exporter only emits End Sites for leaf bones and strips names.
    This function restores the missing named End Sites so that BVH.py counts
    the same number of joints for blend outputs as for GT clips.

    Only End Sites with a ``#name:`` tag and a non-zero offset are added
    (zero-offset End Sites are pruned by BVH.py anyway).
    """
    # ── Step 1: collect named non-trivial End Sites from the GT ref BVH ──────
    extra = {}          # {parent_joint_name: [(es_name, offset_str), ...]}
    jstack = []
    in_es  = False
    es_name, es_offset = None, None

    with open(ref_bvh_path) as f:
        for raw in f:
            s = raw.strip()
            if s == 'MOTION':
                break
            m = re.match(r'(?:ROOT|JOINT)\s+(\S+)', s)
            if m:
                jstack.append(m.group(1))
                in_es = False
                continue
            if s.startswith('End Site'):
                in_es = True
                nm = re.search(r'#name:\s*(\S+)', s)
                es_name   = nm.group(1) if nm else None
                es_offset = None
                continue
            if in_es:
                if s.startswith('OFFSET'):
                    es_offset = s
                elif s == '}':
                    if es_name and es_offset:
                        vals = list(map(float, es_offset.split()[1:]))
                        if any(abs(v) > 1e-6 for v in vals) and jstack:
                            extra.setdefault(jstack[-1], []).append((es_name, es_offset))
                    in_es = False
                    es_name = es_offset = None
                continue
            if s == '}' and jstack:
                jstack.pop()

    if not extra:
        return

    # ── Step 2: patch the exported BVH line-by-line ───────────────────────────
    with open(out_bvh_path) as f:
        lines = f.readlines()

    out   = []
    jstack    = []
    in_hierarchy = True
    in_es_blk = False
    skip_es_blk = False

    for raw in lines:
        s = raw.strip()

        if not in_hierarchy:
            out.append(raw)
            continue

        if s == 'MOTION':
            in_hierarchy = False
            out.append(raw)
            continue

        m = re.match(r'(?:ROOT|JOINT)\s+(\S+)', s)
        if m:
            jstack.append(m.group(1))
            out.append(raw)
            continue

        if s.startswith('End Site'):
            in_es_blk = True
            # If this is an unnamed End Site and the current joint has a named
            # replacement in `extra`, skip this block — the named one will be
            # inserted when the joint's closing `}` is processed.
            skip_es_blk = '#name:' not in s and bool(jstack) and jstack[-1] in extra
            if not skip_es_blk:
                out.append(raw)
            continue

        if s == '}':
            if in_es_blk:
                in_es_blk = False
                if not skip_es_blk:
                    out.append(raw)
                skip_es_blk = False
            elif jstack:
                jname  = jstack[-1]
                indent = '\t' * len(jstack)   # inside the joint's block
                for es_n, es_off in extra.get(jname, []):
                    out.append(f'{indent}End Site #name: {es_n}\n')
                    out.append(f'{indent}{{\n')
                    out.append(f'{indent}\t{es_off}\n')
                    out.append(f'{indent}}}\n')
                jstack.pop()
                out.append(raw)
            else:
                out.append(raw)
            continue

        if not skip_es_blk:
            out.append(raw)

    with open(out_bvh_path, 'w') as f:
        f.writelines(out)


def blend_pair(args, ref_bvh_path: str, tgt_bvh_path: str,
               alpha: float, out_dir: str) -> str:

    clear_scene()

    # ── Import ────────────────────────────────────────────────────────────
    ref_obj, ref_action = import_bvh(ref_bvh_path, scale=args.scale)
    ref_action.name = "Action_REF"
    ref_a_start  = int(ref_action.frame_range[0])
    ref_a_end    = int(ref_action.frame_range[1])
    ref_n_frames = ref_a_end - ref_a_start

    bpy.ops.object.select_all(action="DESELECT")

    tgt_obj, tgt_action = import_bvh(tgt_bvh_path, scale=args.scale)
    tgt_action.name = "Action_TGT"
    tgt_a_start  = int(tgt_action.frame_range[0])
    tgt_a_end    = int(tgt_action.frame_range[1])
    tgt_n_frames = tgt_a_end - tgt_a_start

    # ── Same-skeleton check ───────────────────────────────────────────────
    if not check_skeletons_match(ref_obj, tgt_obj):
        raise ValueError(
            f"Skeleton mismatch:\n  {ref_bvh_path}\n  {tgt_bvh_path}"
        )

    ref_n_joints = len(ref_obj.pose.bones)
    tgt_n_joints = len(tgt_obj.pose.bones)

    # Move tgt_action to ref_obj; delete the separate tgt armature
    bpy.data.objects.remove(tgt_obj, do_unlink=True)

    # ── Frame offsets ─────────────────────────────────────────────────────
    if args.overlap_length >= 0:
        ref_s = 0
        tgt_s = ref_n_frames - args.overlap_length
    else:
        ref_s = args.ref_frame_start
        tgt_s = args.tgt_frame_start

    ref_e        = ref_s + ref_n_frames * args.ref_num_loops
    tgt_e        = tgt_s + tgt_n_frames * args.tgt_num_loops
    total_frames = max(ref_e, tgt_e)

    overlap_s = max(ref_s, tgt_s)
    overlap_e = min(ref_e, tgt_e)
    bake_start = overlap_s if args.overlap_only else 0
    bake_end   = overlap_e if args.overlap_only else total_frames

    # Effective alpha from control_mode
    if args.control_mode == "ref":
        eff_alpha = 0.0
    elif args.control_mode == "tgt":
        eff_alpha = 1.0
    else:
        eff_alpha = alpha

    # ── Set up NLA strips on ref_obj ─────────────────────────────────────
    anim_data = ref_obj.animation_data
    anim_data.action   = None     # detach live action; NLA takes over
    anim_data.use_nla  = True

    for t in list(anim_data.nla_tracks):
        anim_data.nla_tracks.remove(t)

    # Bottom track: reference 
    # REPLACE matches the Oracle's implicit default base behavior
    ref_track = anim_data.nla_tracks.new()
    ref_track.name = "REF"
    ref_strip = ref_track.strips.new("strip_ref", ref_s, ref_action)
    ref_strip.action_frame_start    = ref_a_start
    ref_strip.action_frame_end      = ref_a_end
    ref_strip.frame_start           = ref_s
    ref_strip.frame_end             = ref_e
    ref_strip.repeat                = float(args.ref_num_loops)
    ref_strip.blend_type            = "REPLACE" 
    ref_strip.extrapolation         = "NOTHING"
    # We use F-Curves to emulate the 'blend_out' effect perfectly:
    ref_strip.use_animated_influence = True

    # Top track: target 
    # COMBINE matches the Oracle's explicit setting
    tgt_track = anim_data.nla_tracks.new()
    tgt_track.name = "TGT"
    tgt_strip = tgt_track.strips.new("strip_tgt", tgt_s, tgt_action)
    tgt_strip.action_frame_start    = tgt_a_start
    tgt_strip.action_frame_end      = tgt_a_end
    tgt_strip.frame_start           = tgt_s
    tgt_strip.frame_end             = tgt_e
    tgt_strip.repeat                = float(args.tgt_num_loops)
    tgt_strip.blend_type            = "COMBINE"
    tgt_strip.extrapolation         = "NOTHING"
    # We use F-Curves to emulate the 'blend_in' effect perfectly:
    tgt_strip.use_animated_influence = True

    # ── Keyframe strip influences ─────────────────────────────────────────
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end   = total_frames

    bpy.context.view_layer.objects.active = ref_obj
    ref_obj.select_set(True)

    def insert_strip_key(strip, frame, value):
        scene.frame_set(frame)
        strip.influence = value
        strip.keyframe_insert(data_path="influence", frame=frame)

    # Build influence keypoints for BOTH strips to achieve the crossfade
    ref_keys, tgt_keys = build_influence_keypoints(
        ref_s, ref_e, tgt_s, tgt_e,
        eff_alpha, args.blend_schedule, args.ease_slope
    )

    for frame, value in ref_keys:
        insert_strip_key(ref_strip, frame, value)
    for frame, value in tgt_keys:
        insert_strip_key(tgt_strip, frame, value)

    interp = "LINEAR" if args.blend_schedule in ("static", "linear") else "BEZIER"
    _set_fcurve_interpolation(ref_strip, interp)
    _set_fcurve_interpolation(tgt_strip, interp)
    if args.blend_schedule == "ease":
        _set_bezier_auto_handles(ref_strip)
        _set_bezier_auto_handles(tgt_strip)

    # ── Bake NLA (per-frame evaluation) ────────────────────────────────────
    import mathutils
    sampled = {}
    for out_f in range(bake_start, bake_end + 1):
        scene.frame_set(out_f)
        bpy.context.view_layer.update()
        frame_data = {}
        for bone in ref_obj.pose.bones:
            frame_data[bone.name] = (
                bone.location.copy(),
                bone.rotation_quaternion.copy(),
                mathutils.Euler(bone.rotation_euler[:], bone.rotation_euler.order),
                bone.scale.copy(),
                bone.rotation_mode,
            )
        sampled[out_f] = frame_data

    # Write captured transforms into a clean action (NLA off)
    baked_action = bpy.data.actions.new("Action_BAKED")
    anim_data.use_nla = False
    anim_data.action  = baked_action

    for out_f, frame_data in sampled.items():
        baked_f = out_f - bake_start  # remap to 0-based when overlap_only
        scene.frame_set(baked_f)
        for bone in ref_obj.pose.bones:
            loc, rot_q, rot_e, scale, rot_mode = frame_data[bone.name]
            bone.rotation_mode = rot_mode
            bone.location      = loc
            bone.scale         = scale
            if rot_mode == "QUATERNION":
                bone.rotation_quaternion = rot_q
                bone.keyframe_insert("rotation_quaternion", frame=baked_f)
            else:
                bone.rotation_euler = rot_e
                bone.keyframe_insert("rotation_euler", frame=baked_f)
            if bone.parent is None:  # root: keyframe location to preserve global translation
                bone.keyframe_insert("location", frame=baked_f)

    # ── Export BVH ────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    ref_stem = Path(ref_bvh_path).stem
    tgt_stem = Path(tgt_bvh_path).stem

    if args.control_mode == "ref":
        blend_tag = "ref"
    elif args.control_mode == "tgt":
        blend_tag = "tgt"
    elif args.overlap_only:
        _sched = args.blend_schedule
        if _sched == "ease" and args.ease_slope != 1.0:
            _sched = f"ease_slope{args.ease_slope:g}"
        if _sched == "static":
            blend_tag = f"intersection_{_sched}_alpha{eff_alpha:.2f}".rstrip("0").rstrip(".")
        else:
            blend_tag = f"intersection_{_sched}"
    elif args.overlap_length >= 0:
        _sched = args.blend_schedule
        if _sched == "ease" and args.ease_slope != 1.0:
            _sched = f"ease_slope{args.ease_slope:g}"
        blend_tag = f"temporal_{_sched}"
    elif args.blend_schedule == "static":
        blend_tag = f"alpha{eff_alpha:.2f}".rstrip("0").rstrip(".")
    else:
        blend_tag = args.blend_schedule

    out_name = f"{ref_stem}___{tgt_stem}___{blend_tag}.bvh"
    out_bvh  = os.path.join(out_dir, out_name)

    bpy.ops.export_anim.bvh(
        filepath=out_bvh,
        frame_start=0,
        frame_end=bake_end - bake_start - 1,
        global_scale=1.0 / args.scale,
        rotate_mode="NATIVE",
        root_transform_only=True,   # root → CHANNELS 6, all others → CHANNELS 3 (matches GT)
    )
    # Restore named End Sites (blender doesn't natively allow for names on endsite).
    _patch_end_sites(out_bvh, ref_bvh_path)

    # ── Metadata JSON ─────────────────────────────────────────────────────
    with open(out_bvh.replace(".bvh", ".json"), "w") as f:
        json.dump({
            "samples": {
                "ref":        ref_stem,
                "ref_path":   ref_bvh_path,
                "ref_object": ref_stem.split("_")[0] + ("_m" if ref_stem.split("_")[1:2] == ["m"] else ""),
                "tgt":        tgt_stem,
                "tgt_path":   tgt_bvh_path,
                "tgt_object": tgt_stem.split("_")[0] + ("_m" if tgt_stem.split("_")[1:2] == ["m"] else ""),
            },
            "regions": {
                "ref_s":         ref_s,
                "ref_e":         ref_e,
                "tgt_s":         tgt_s,
                "tgt_e":         tgt_e,
                "overlap_s":     overlap_s,
                "overlap_e":     overlap_e,
                "same_skeleton": True,
                "ref_n_joints":  ref_n_joints,
                "tgt_n_joints":  tgt_n_joints,
            },
            "blend": {
                "control_mode":       args.control_mode,
                "alpha":              eff_alpha,
                "blend_schedule":     args.blend_schedule,
                "ease_slope":         args.ease_slope,
                "overlap_length":     args.overlap_length,
                "overlap_only":       args.overlap_only,
                "ref_frame_start":    ref_s,
                "tgt_frame_start":    tgt_s,
                "ref_num_loops":      args.ref_num_loops,
                "tgt_num_loops":      args.tgt_num_loops,
                "interpolation_mode": None,
            },
            "method": "blender_nla",
            "output": out_bvh,
        }, f, indent=4)

    print(f"[nla_blend] Saved: {out_bvh}")
    return out_bvh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.ref_samples and args.tgt_samples:
        bvh_dir = args.bvh_dir or os.path.join(
            os.path.dirname(args.dataset_npy_dir), "bvhs"
        )
        with open(args.ref_samples) as f:
            ref_list = [os.path.join(bvh_dir, f"{l.strip()}.bvh")
                        for l in f if l.strip()]
        with open(args.tgt_samples) as f:
            tgt_list = [os.path.join(bvh_dir, f"{l.strip()}.bvh")
                        for l in f if l.strip()]
    else:
        assert args.ref_motion and args.tgt_motion, \
            "Provide --ref_motion + --tgt_motion  or  --ref_samples + --tgt_samples"
        ref_list = [args.ref_motion]
        tgt_list = [args.tgt_motion]

    assert len(ref_list) == len(tgt_list)

    if args.control_mode == "ref":
        alpha_values = [0.0]
    elif args.control_mode == "tgt":
        alpha_values = [1.0]
    else:
        alpha_values = args.alpha_values

    out_path = args.output_dir
    
    if args.ref_samples and args.tgt_samples:
        _ref_dir    = os.path.dirname(os.path.abspath(args.ref_samples))
        _category   = os.path.basename(_ref_dir)
        _bench_type = os.path.basename(os.path.dirname(_ref_dir))
        _skel_mode  = _bench_type if _bench_type in ("in_skel", "x_skel") else "bench"

        if args.control_mode == "ref":
            _desc = "ref"
        elif args.control_mode == "tgt":
            _desc = "tgt"
        elif args.overlap_only:
            _sched = args.blend_schedule
            if _sched == "ease" and args.ease_slope != 1.0:
                _sched = f"ease_slope{args.ease_slope:g}"
            if _sched == "static" and len(args.alpha_values) == 1:
                _a = args.alpha_values[0]
                _desc = f"intersection_{_sched}_alpha{_a:.2f}".rstrip("0").rstrip(".")
            else:
                _desc = f"intersection_{_sched}"
        elif args.overlap_length >= 0:
            _sched = args.blend_schedule
            if _sched == "ease" and args.ease_slope != 1.0:
                _sched = f"ease_slope{args.ease_slope:g}"
            _desc = f"temporal_{_sched}"
        elif args.blend_schedule == "static" and len(args.alpha_values) == 1:
            _a = args.alpha_values[0]
            _desc = f"alpha{_a:.2f}".rstrip("0").rstrip(".")
        else:
            _desc = args.blend_schedule

        out_path = os.path.join(out_path, f"bench-{_skel_mode}-{_category}-{_desc}")

    os.makedirs(out_path, exist_ok=True)

    for ref_bvh, tgt_bvh in zip(ref_list, tgt_list):
        print(f"\nProcessing:\n  REF: {ref_bvh}\n  TGT: {tgt_bvh}")
        for alpha in alpha_values:
            print(f"  alpha={alpha}  schedule={args.blend_schedule}")
            try:
                blend_pair(args, ref_bvh, tgt_bvh, alpha, out_path)
            except ValueError as e:
                print(f"[nla_blend] SKIPPED — {e}")

if __name__ == "__main__":
    main()