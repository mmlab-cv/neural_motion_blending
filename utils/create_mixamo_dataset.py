"""
Preprocessing script for the Mixamo dataset.

Mirrors the Truebones pipeline (utils/create_dataset.py) but operates over the
Mixamo BVH collection at dataset/MixamoBVH/.  Output is written to
dataset/mixamo_processed/ with the identical directory structure used by
truebones_processed:

    dataset/mixamo_processed/
        motions/         *.npy  (Frames-1, Joints, 13) feature tensors
        bvhs/            *.bvh  normalised BVH files
        animations/      *.mp4  sanity-check visualisations
        cond.npy         per-character conditioning dict
        metadata.txt
        positions_error_rate.txt

Character directories ending with _m (homeomorphic variants) are skipped.
Train/test splits defined in dataset/mixamo_processed/filelist/ are respected:
only the BVH files listed there are processed; the character-named BVH
(e.g. Abe/Abe.bvh) is always used as the T-pose reference.

Run from the mmix root:
    python utils/create_mixamo_dataset.py
"""

import os
import shutil
import tempfile
import numpy as np
from os.path import join as pjoin

from data_loaders.truebones.truebones_utils.motion_process import process_object
from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_DIR, ANIMATIONS_DIR, BVHS_DIR, MAX_PATH_LEN,
)

SKIP_CHARACTERS = ['Parasite_m']

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MIXAMO_RAW_DIR  = "dataset/MixamoBVH"
MIXAMO_OUT_DIR  = "dataset/mixamo_processed"
FILELIST_DIR    = pjoin(MIXAMO_OUT_DIR, "filelist")

# ---------------------------------------------------------------------------
# Face joints: four landmarks defining skeleton orientation.
# Regular Mixamo characters use the "mixamorig:" prefix; _m variants use plain
# names (their rig is the homeomorphic variant without the prefix).
# ---------------------------------------------------------------------------

def mixamo_jointname_cleanup(name: str) -> str:
    return str.split(name, ":")[-1]

MIXAMO_FACE_JOINTS = [
    "RightUpLeg",   # right hip
    "LeftUpLeg",    # left hip
    "RightArm",     # right shoulder
    "LeftArm",      # left shoulder
]


def get_face_joints(char_name: str) -> list:

    special_character_idmap = {
        'James': 9, 'Joe': 7, 'Man': 1, 'Racer': 6, 'Timmy': 6,
        'Penguin': ['rThigh', 'lThigh', 'rArm', 'lArm']
    }

    if char_name in special_character_idmap:
        joint_id = special_character_idmap[char_name]
        if char_name != 'Penguin':
            return [f"mixamorig{joint_id}:{n}" for n in MIXAMO_FACE_JOINTS]
        return joint_id
    
    elif not char_name.endswith("_m"):
        return [f"mixamorig:{n}" for n in MIXAMO_FACE_JOINTS]

    return MIXAMO_FACE_JOINTS


def _collect_split_files(char_name: str, splits: list) -> list:
    """
    Return BVH filenames for *char_name* from the requested *splits*
    (any subset of ["train", "test"]).

    Returns an empty list if no filelist is found for any of the splits.
    """
    key = char_name.lower()
    files = set()
    for split in splits:
        candidate = pjoin(FILELIST_DIR, split, f"{split}_{key}.txt")
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        files.add(line)
    return sorted(files)


def _iter_characters(splits: list) -> list:
    """
    Yield (char_name, bvh_dir, split_files, tpos_bvh_path) for every valid
    Mixamo character:
      - special marker directories (mean_var, std_bvhs) are skipped
      - characters with no filelist entry are skipped
      - the T-pose BVH is the file named <CharName>.bvh inside the character
        directory (which encodes the Mixamo T-pose in its joint hierarchy)
    """
    skip_dirs = ["mean_var", "std_bvhs"]
    entries = []
    for char_name in sorted(os.listdir(MIXAMO_RAW_DIR)):
        
        if char_name in skip_dirs + SKIP_CHARACTERS:
            continue

        bvh_dir = pjoin(MIXAMO_RAW_DIR, char_name)
        assert os.path.isdir(bvh_dir), f"Expected directory not found for {char_name}: {bvh_dir}"

        filelist_name = char_name[:-2] if char_name.endswith("_m") else char_name
        stripped_char_name = char_name[:-2] if char_name.endswith("_m") else char_name
        
        split_files = _collect_split_files(filelist_name, splits)
        if not split_files:
            print(f"  [skip] {char_name}: no filelist found for splits {', '.join(splits)}")
            continue

        # The eponymous BVH encodes the character's T-pose
        tpos_bvh = pjoin(bvh_dir, f"{stripped_char_name}.bvh")
        assert os.path.exists(tpos_bvh), f"T-pose BVH not found for {char_name} at expected path: {tpos_bvh}"
        entries.append((char_name, bvh_dir, split_files, tpos_bvh, get_face_joints(char_name)))
        
    return entries


def _make_temp_bvh_dir(char_name: str, bvh_dir: str, split_files: list, tpos_bvh: str):
    """
    Build a temporary directory of symlinks containing only the split-listed
    BVH files plus the T-pose BVH, so that process_object does not accidentally
    pick up files outside the train/test split.

    process_object discovers BVH files by listing the directory and then tries
    to remove `t_pos_path` from that list by path equality — so the T-pose
    symlink must be present in the temp dir and its path returned.

    Returns (tmp_dir, tmp_tpos_path).
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"mixamo_{char_name}_")

    # Symlink all split BVH files
    for fname in split_files:
        src = pjoin(bvh_dir, fname)
        dst = pjoin(tmp_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    # Symlink the T-pose BVH and return its path inside the temp dir
    tmp_tpos_path = None
    if tpos_bvh is not None and os.path.isfile(tpos_bvh):
        tpos_fname = os.path.basename(tpos_bvh)
        dst_tpos = pjoin(tmp_dir, tpos_fname)
        if not os.path.exists(dst_tpos):
            os.symlink(os.path.abspath(tpos_bvh), dst_tpos)
        tmp_tpos_path = dst_tpos

    return tmp_dir, tmp_tpos_path


def recompute_stats():
    """
    Recompute mean/std for every character in cond.npy from the already-processed
    .npy files in motions/, then write the updated cond.npy back to disk.
    Stats are always computed from ALL available motions for each character,
    regardless of which split was used when the motions were originally generated.
    """
    cond_path = pjoin(MIXAMO_OUT_DIR, "cond.npy")
    if not os.path.isfile(cond_path):
        raise FileNotFoundError(f"cond.npy not found at {cond_path} — run the full pipeline first.")

    cond = np.load(cond_path, allow_pickle=True).item()
    motion_dir = pjoin(MIXAMO_OUT_DIR, MOTION_DIR)
    all_npy = [f for f in os.listdir(motion_dir) if f.endswith(".npy")]

    from data_loaders.truebones.truebones_utils.motion_process import get_mean_std

    updated = 0
    for char_name in sorted(cond.keys()):
        prefix = f"{char_name}_"
        char_files = [f for f in all_npy if f.startswith(prefix)
                      and not f.startswith(f"{char_name}_m_")]
        assert char_files, f"  [skip] {char_name}: no .npy files found in motions/"
        tensors = np.concatenate(
            [np.load(pjoin(motion_dir, f)) for f in char_files], axis=0
        )
        mean, std = get_mean_std(tensors)
        cond[char_name]["mean"] = mean
        cond[char_name]["std"]  = std
        print(f"  {char_name}: {len(char_files)} clips, {tensors.shape[0]} frames")
        updated += 1

    np.save(cond_path, cond)
    print(f"\nUpdated stats for {updated}/{len(cond)} characters -> {cond_path}")


def create_mixamo_dataset(dry_run: bool = False, splits: list = None,
                          max_animations: int = 3, only_chars: list = None):
    if splits is None:
        splits = ["train", "test"]

    os.makedirs(pjoin(MIXAMO_OUT_DIR, MOTION_DIR),     exist_ok=True)
    os.makedirs(pjoin(MIXAMO_OUT_DIR, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(MIXAMO_OUT_DIR, BVHS_DIR),       exist_ok=True)

    characters = _iter_characters(splits)
    if only_chars:
        only_set   = set(only_chars)
        characters = [c for c in characters if c[0] in only_set]
    print(f"Found {len(characters)} characters to process."
          + f" [splits: {', '.join(splits)}]"
          + (f" [chars: {', '.join(c[0] for c in characters)}]" if only_chars else "")
          + (" [DRY RUN: 1 motion per character]" if dry_run else ""))

    files_counter  = 0
    frames_counter = 0
    max_joints     = 143 # As Truebones
    objects_counter        = {}
    squared_positions_error = {}
    cond                   = {}

    for char_name, bvh_dir, split_files, tpos_bvh, face_joints in characters:
        active_files = split_files[:3] if dry_run else split_files

        print(f"\n{'='*60}")
        print(f"Processing: {char_name}  ({len(active_files)}/{len(split_files)} motions)")

        # Build a temp dir containing only the active BVH files so that
        # process_object does not accidentally include T-pose or _m files.
        # The T-pose symlink is also placed there so path equality check inside
        # process_object can correctly remove it from the motion files list.
        tmp_dir, tmp_tpos = _make_temp_bvh_dir(char_name, bvh_dir, active_files, tpos_bvh)

        cur_counter = files_counter
        try:
            # Monkey-patch the plot function so only the first max_animations
            # clips per character actually render an MP4; the rest are no-ops.
            import data_loaders.truebones.truebones_utils.motion_process as mp_module
            _real_plot = mp_module.plot_general_skeleton_3d_motion
            _anim_count = [0]

            def _selective_plot(*args, **kwargs):
                _anim_count[0] += 1
                if _anim_count[0] <= max_animations:
                    _real_plot(*args, **kwargs)

            mp_module.plot_general_skeleton_3d_motion = _selective_plot
            try:
                files_counter, frames_counter, max_joints, object_cond = process_object(
                    object_type=char_name,
                    files_counter=files_counter,
                    frames_counter=frames_counter,
                    max_joints=max_joints,
                    squared_positions_error=squared_positions_error,
                    save_dir=MIXAMO_OUT_DIR,
                    face_joints=face_joints,
                    bvhs_dir=tmp_dir,
                    t_pos_path=tmp_tpos
                )
            finally:
                mp_module.plot_general_skeleton_3d_motion = _real_plot

            # Strip the "mixamorig:" prefix that Mixamo embeds in every joint
            # name.  The prefix is shared across all joints and carries no
            # semantic information, so it would only dilute T5 embeddings.
            if 'joints_names' in object_cond:
                object_cond['joints_names'] = [
                    mixamo_jointname_cleanup(n) for n in object_cond['joints_names']
                ]
            cond[char_name] = object_cond
            objects_counter[char_name] = files_counter - cur_counter
            print(f"  -> {objects_counter[char_name]} clips written")
        except Exception as exc:
            raise RuntimeError(f"  [ERROR] {char_name}: {exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    from data_loaders.truebones.truebones_utils.motion_process import get_mean_std
    motion_dir = pjoin(MIXAMO_OUT_DIR, MOTION_DIR)
    all_npy = [f for f in os.listdir(motion_dir) if f.endswith(".npy")]
    for char_name in list(cond.keys()):
        prefix = f"{char_name}_"
        char_files = [f for f in all_npy if f.startswith(prefix)
                      and not f.startswith(f"{char_name}_m_")]
        assert char_files, f"  [skip] {char_name}: no .npy files found in motions/"
        
        tensors = np.concatenate([np.load(pjoin(motion_dir, f)) for f in char_files], axis=0)
        cond[char_name]["mean"], cond[char_name]["std"] = get_mean_std(tensors)

    # Merge with existing cond.npy so a partial re-run doesn't wipe other characters
    cond_path = pjoin(MIXAMO_OUT_DIR, "cond.npy")
    if os.path.isfile(cond_path) and only_chars:
        existing = np.load(cond_path, allow_pickle=True).item()
        existing.update(cond)
        cond = existing
    np.save(cond_path, cond)

    total_clips = sum(objects_counter.values())
    print(f"\nTotal clips: {files_counter}, Frames: {frames_counter}, "
          f"Duration: {frames_counter / 12.5 / 60:.1f}m")
    print(f"Max joints: {max_joints}")

    with open(pjoin(MIXAMO_OUT_DIR, "metadata.txt"), "w") as f:
        f.write(f"max joints: {max_joints}\n")
        f.write(f"total frames: {frames_counter}\n")
        f.write(f"duration: {frames_counter / 12.5 / 60:.1f}\n")
        f.write(f"~~~~ objects_counts - Total: {total_clips} ~~~~\n")
        for obj, cnt in objects_counter.items():
            f.write(f"{obj}: {cnt}\n")

    with open(pjoin(MIXAMO_OUT_DIR, "positions_error_rate.txt"), "w") as f:
        f.write("Position squared error per bvh file:\n")
        for fpath, err in squared_positions_error.items():
            f.write(f"{fpath}: {err}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Mixamo BVH dataset.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 1 motion per character to verify the pipeline end-to-end.",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="both",
        help="Which split to process (default: both).",
    )
    parser.add_argument(
        "--max-animations", type=int, default=0,
        help="Max MP4 animations to keep per character (default: 3). Pass 0 to disable all.",
    )
    parser.add_argument(
        "--get-stats-only", action="store_true",
        help="Skip motion conversion entirely; recompute mean/std in cond.npy from "
             "already-processed motions/ files and exit.",
    )
    parser.add_argument(
        "--chars", nargs="+", default=None, metavar="CHAR",
        help="Only process these characters (e.g. --chars Vampire_m Parasite_m). "
             "Merges results into the existing cond.npy instead of overwriting it.",
    )
    args = parser.parse_args()

    if args.get_stats_only:
        recompute_stats()
        return

    splits = ["train", "test"] if args.split == "both" else [args.split]
    create_mixamo_dataset(dry_run=args.dry_run, splits=splits,
                          max_animations=args.max_animations, only_chars=args.chars)


if __name__ == "__main__":
    main()
