"""
build_tpose_ref.py — Placeholder --ref_motion builder for retargeting with no motion clip on disk
(see docs/SAMPLE.md). Builds a T-pose-only .npy for a character straight from cond.npy, valid as
--ref_motion for sample.mix --control_mode tgt, which discards ref content anyway.

Usage:
    python -m utils.build_tpose_ref --list
    python -m utils.build_tpose_ref --object_type Alligator
"""

import argparse
import os

import numpy as np

from data_loaders.truebones.truebones_utils.get_opt import get_opt


def list_characters(cond_dict):
    for name in sorted(cond_dict):
        print(name)


def build_tpose_ref(cond_dict, object_type, num_frames, output_dir):
    if object_type not in cond_dict:
        raise ValueError(
            f"Unknown object_type {object_type!r}. Run with --list to see available characters."
        )
    tpos = np.asarray(cond_dict[object_type]['tpos_first_frame'])  # (n_joints, feature_len)
    placeholder = np.tile(tpos[None], (num_frames, 1, 1))          # (num_frames, n_joints, feature_len)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{object_type}_tpose.npy')
    np.save(out_path, placeholder)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['truebones', 'mixamo'], default='truebones',
                        help="Dataset to read cond.npy from. Ignored if --cond_path is set.")
    parser.add_argument('--cond_path', default='', type=str,
                        help="Path to a specific cond.npy. Overrides --dataset.")
    parser.add_argument('--list', action='store_true',
                        help="List available characters and exit.")
    parser.add_argument('--object_type', default='', type=str,
                        help="Character to build a placeholder for (must be a key in cond.npy).")
    parser.add_argument('--num_frames', default=60, type=int,
                        help="Frame count for the placeholder clip (content is discarded, so any length works).")
    parser.add_argument('--output_dir', default='.', type=str,
                        help="Directory to write <object_type>_tpose.npy into.")
    args = parser.parse_args()

    if not args.list and not args.object_type:
        parser.error("Pass --list to see available characters, or --object_type to build a placeholder.")

    cond_file = args.cond_path or get_opt(device=0, dataset=args.dataset).cond_file
    if not os.path.isfile(cond_file):
        raise FileNotFoundError(
            f"cond.npy not found at {cond_file} — process the dataset first (see docs/DATA.md)."
        )
    cond_dict = np.load(cond_file, allow_pickle=True).item()

    if args.list:
        list_characters(cond_dict)
        return

    out_path = build_tpose_ref(cond_dict, args.object_type, args.num_frames, args.output_dir)
    print(f"Wrote {args.num_frames}-frame T-pose placeholder for {args.object_type!r} -> {out_path}")
    print(f"Use it as --ref_motion {out_path} with --control_mode tgt.")


if __name__ == '__main__':
    main()
