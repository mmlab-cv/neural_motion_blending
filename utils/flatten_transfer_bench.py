#!/usr/bin/env python3
"""
Flatten the nested transfer-bench output directory into the flat layout expected by
eval/eval_truebones_blend.py, and generate the required JSON metadata files.

The nested structure produced by the other machine is:
    <nested_dir>/<TGT_from_SRC>/<TGT___motionA__src_SRC___motionB>/run_N.bvh

The flat layout expected by the eval script is:
    <flat_dir>/<stem>.bvh      (or symlinks)
    <flat_dir>/<stem>.json     (metadata file)

where <stem> encodes the pair and repetition index.

For the transfer benchmark the 'blend' regions are:
    ref_s = 0,  ref_e = min(ref_frames, tgt_frames)  (= overlap_e)
    tgt_s = 0,  tgt_e = tgt_frames
    overlap_s = 0,  overlap_e = min(ref_frames, tgt_frames)
    same_skeleton = False   (always x_skel for transfer)

Usage:
    python3 utils/flatten_transfer_bench.py \\
        --nested_dir  save/motion2motion/bench-x_skel-walk_walk-transfer \\
        --flat_dir    save/motion2motion/bench-x_skel-walk_walk-transfer-flat \\
        --gt_dir      dataset/truebones/zoo/truebones_processed/bvhs \\
        [--symlink]   # use symlinks instead of copies (saves disk space)
"""

import argparse
import json
import os
import shutil
from utils.misc import char_from_npy_stem


def get_bvh_n_frames(path: str) -> int:
    """Return the number of frames in a BVH file without importing the full stack."""
    import BVH  # type: ignore
    anim, _, _ = BVH.load(path)
    return int(anim.rotations.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten nested transfer-bench BVH output and generate JSON metadata.",
    )
    parser.add_argument("--nested_dir", required=True,
                        help="Root of the nested output directory.")
    parser.add_argument("--flat_dir", required=True,
                        help="Destination flat directory (created if absent).")
    parser.add_argument("--gt_dir", required=True,
                        help="GT BVH directory (e.g. .../truebones_processed/bvhs).")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink BVH files instead of copying them.")
    args = parser.parse_args()

    nested_dir = os.path.abspath(args.nested_dir)
    flat_dir   = os.path.abspath(args.flat_dir)
    gt_dir     = os.path.abspath(args.gt_dir)

    os.makedirs(flat_dir, exist_ok=True)

    n_written  = 0
    n_skipped  = 0

    for pair_dir in sorted(os.listdir(nested_dir)):
        pair_path = os.path.join(nested_dir, pair_dir)
        if not os.path.isdir(pair_path):
            continue

        for inner_dir in sorted(os.listdir(pair_path)):
            inner_path = os.path.join(pair_path, inner_dir)
            if not os.path.isdir(inner_path):
                continue

            # Parse ref/tgt names from the inner folder name.
            # Expected pattern:  <REF_NAME>__src_<TGT_NAME>
            if "__src_" not in inner_dir:
                print(f"  SKIP (unexpected inner dir name): {inner_dir}")
                n_skipped += 1
                continue

            ref_name, tgt_name = inner_dir.split("__src_", 1)
            ref_char = ref_name.split("___")[0] if "___" in ref_name else char_from_npy_stem(ref_name)

            # Resolve GT paths.
            ref_gt_path = os.path.join(gt_dir, ref_name + ".bvh")
            tgt_gt_path = os.path.join(gt_dir, tgt_name + ".bvh")

            if not os.path.exists(ref_gt_path):
                print(f"  SKIP (GT ref not found): {ref_gt_path}")
                n_skipped += 1
                continue
            if not os.path.exists(tgt_gt_path):
                print(f"  SKIP (GT tgt not found): {tgt_gt_path}")
                n_skipped += 1
                continue

            ref_frames = get_bvh_n_frames(ref_gt_path)
            tgt_frames = get_bvh_n_frames(tgt_gt_path)
            overlap_e  = min(ref_frames, tgt_frames)

            regions = {
                "ref_s":         0,
                "ref_e":         overlap_e,
                "tgt_s":         0,
                "tgt_e":         tgt_frames,
                "overlap_s":     0,
                "overlap_e":     overlap_e,
                "same_skeleton": False,
            }

            # Process each run_N.bvh file.
            bvh_files = sorted(
                f for f in os.listdir(inner_path) if f.endswith(".bvh")
            )
            for bvh_file in bvh_files:
                run_idx  = os.path.splitext(bvh_file)[0]  # e.g. "run_0"
                stem     = f"{inner_dir}__{run_idx}"       # unique flat stem
                dst_bvh  = os.path.join(flat_dir, stem + ".bvh")
                dst_json = os.path.join(flat_dir, stem + ".json")

                # BVH: symlink or copy.
                src_bvh = os.path.join(inner_path, bvh_file)
                if not os.path.exists(dst_bvh):
                    if args.symlink:
                        os.symlink(src_bvh, dst_bvh)
                    else:
                        shutil.copy2(src_bvh, dst_bvh)

                # JSON metadata.
                if not os.path.exists(dst_json):
                    meta = {
                        "samples": {
                            "ref":        ref_name,
                            "ref_path":   ref_gt_path,
                            "ref_object": ref_char,
                            "tgt":        tgt_name,
                            "tgt_path":   tgt_gt_path,
                            "tgt_object": tgt_name.split("___")[0] if "___" in tgt_name else char_from_npy_stem(tgt_name),
                        },
                        "regions": regions,
                        "blend": {
                            "control_mode": "tgt",
                            "alpha": 1.0,
                        },
                        "method": "mmix",
                    }
                    with open(dst_json, "w") as f:
                        json.dump(meta, f, indent=2)

                n_written += 1
                print(f"  {stem}")

    print(f"\nDone — {n_written} entries written, {n_skipped} skipped.")
    print(f"Flat dir: {flat_dir}")


if __name__ == "__main__":
    main()
