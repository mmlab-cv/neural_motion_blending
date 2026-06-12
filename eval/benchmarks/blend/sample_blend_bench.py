#!/usr/bin/env python3
"""
Generate blend benchmark ref.txt / tgt.txt pairs.

Usage examples:
    # in-skeleton Walk→Run, 3 samples per character
    python sample_blend_bench.py --mode in_skel --category Walk_Run --n_per_skel 3

    # cross-skeleton Idle→Walk, 2 samples per character, custom output dir
    python sample_blend_bench.py --mode x_skel --category Idle_Walk --n_per_skel 2 --output_dir ./x_skel/idle_walk

Category names match the .txt files in the same directory (walk.txt, run.txt, idle.txt, …).
Output is written to <mode>/<category_lower>/ref.txt and tgt.txt by default.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np

CATEGORY_DIR = os.path.dirname(os.path.abspath(__file__))


def filter_by_length(samples: list[str], dataset_dir: str, min_length: int) -> list[str]:
    """
    Return only samples whose .npy file has at least min_length frames.
    Skips samples whose file cannot be found (with a warning).
    """
    kept = []
    for name in samples:
        path = os.path.join(dataset_dir, f"{name}.npy")
        if not os.path.exists(path):
            print(f"  [filter_by_length] WARNING: {path} not found, skipping.")
            continue
        try:
            n_frames = int(np.load(path, mmap_mode='r').shape[0])
        except Exception as e:
            print(f"  [filter_by_length] WARNING: could not read {path}: {e}")
            continue
        if n_frames >= min_length:
            kept.append(name)
    removed = len(samples) - len(kept)
    if removed:
        print(f"  [filter_by_length] Removed {removed}/{len(samples)} samples shorter than {min_length} frames.")
    return kept


def load_category(name: str) -> list[str]:
    """Load sample names from a category file, e.g. 'walk' → walk.txt."""
    path = os.path.join(CATEGORY_DIR, f"{name.lower()}.txt")
    if not os.path.exists(path):
        available = [
            f[:-4] for f in os.listdir(CATEGORY_DIR)
            if f.endswith('.txt') and f != 'all.txt'
        ]
        raise FileNotFoundError(
            f"Category file not found: {path}\n"
            f"Available categories: {sorted(available)}"
        )
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract_character(sample_name: str) -> str:
    """
    Extract the skeleton/character identifier from a sample name.

    Naming conventions found in the dataset:
      - Alligator___Walk1_19   → 'Alligator'   (triple-underscore separator)
      - Fox_-_Walk_372         → 'Fox'          (dash separator)
      - Cat_CAT_Walk_197       → 'Cat_CAT'      (Name_ABBREV_Action_ID)
      - Puppy_Puppy_Walk_669   → 'Puppy_Puppy'  (repeated name + Action + ID)
    """
    if '___' in sample_name:
        return sample_name.split('___')[0]
    if '_-_' in sample_name:
        return sample_name.split('_-_')[0]
    # Fallback: strip trailing _ID and _ActionWord
    parts = sample_name.split('_')
    if len(parts) > 2:
        return '_'.join(parts[:-2])
    return parts[0]


def group_by_character(samples: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[extract_character(s)].append(s)
    return groups


def sample_in_skel(
    ref_samples: list[str],
    tgt_samples: list[str],
    n_per_skel: int,
    rng: random.Random,
    no_self_pairs: bool = False,
) -> tuple[list[str], list[str]]:
    """
    In-skeleton pairing: ref and tgt come from the *same* character.
    For every character that has at least one motion in each category,
    pick up to n_per_skel ref motions and pair each with a randomly chosen
    tgt motion from the same character.
    """
    ref_groups = group_by_character(ref_samples)
    tgt_groups = group_by_character(tgt_samples)
    common_chars = sorted(set(ref_groups) & set(tgt_groups))

    seen: set[tuple[str, str]] = set()
    refs, tgts = [], []
    for char in common_chars:
        r_pool = ref_groups[char]
        t_pool = tgt_groups[char]
        chosen_refs = rng.sample(r_pool, min(n_per_skel, len(r_pool)))
        for r in chosen_refs:
            t_candidates = [t for t in t_pool if (r, t) not in seen]
            if not t_candidates:
                continue
            # Prefer non-self pairs; fall back to self only if no other option
            # (or if --allow_self_pairs is set).
            non_self = [t for t in t_candidates if t != r]
            if non_self:
                t = rng.choice(non_self)
            elif not no_self_pairs:
                t = rng.choice(t_candidates)  # only self left, allow it
            else:
                continue  # skip: self-pairs explicitly forbidden
            seen.add((r, t))
            refs.append(r)
            tgts.append(t)
    return refs, tgts


def sample_x_skel(
    ref_samples: list[str],
    tgt_samples: list[str],
    n_per_skel: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """
    Cross-skeleton pairing: ref and tgt come from *different* characters.
    For every character in the ref pool, pick up to n_per_skel ref motions
    and pair each with a tgt motion drawn from any other character.
    """
    ref_groups = group_by_character(ref_samples)
    tgt_groups = group_by_character(tgt_samples)

    seen: set[tuple[str, str]] = set()
    refs, tgts = [], []
    for char in sorted(ref_groups):
        r_pool = ref_groups[char]
        chosen_refs = rng.sample(r_pool, min(n_per_skel, len(r_pool)))
        other_tgts = [s for c, pool in tgt_groups.items() if c != char for s in pool]
        if not other_tgts:
            continue
        for r in chosen_refs:
            t_candidates = [t for t in other_tgts if (r, t) not in seen]
            if not t_candidates:
                continue
            t = rng.choice(t_candidates)
            seen.add((r, t))
            refs.append(r)
            tgts.append(t)
    return refs, tgts


def parse_category(category_str: str) -> tuple[str, str]:
    """
    Parse 'Walk_Run' into ('walk', 'run').
    The split point is the first '_', so 'SlowWalk_Run' → ('slowwalk', 'run').
    """
    parts = category_str.split('_', 1)
    if len(parts) != 2:
        raise ValueError(
            f"--category must be 'RefCat_TgtCat' (e.g. 'Walk_Run'), got: {category_str!r}"
        )
    return parts[0].lower(), parts[1].lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly populate a blend benchmark (ref.txt / tgt.txt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['in_skel', 'x_skel'], required=True,
        help="'in_skel': ref and tgt share the same skeleton; "
             "'x_skel': ref and tgt use different skeletons.",
    )
    parser.add_argument(
        '--category', required=True, metavar='RefCat_TgtCat',
        help="Two semantic categories separated by '_', e.g. 'Walk_Run' or 'Idle_Walk'. "
             "Each name must match a <name>.txt file in the blend directory.",
    )
    parser.add_argument(
        '--n_per_skel', type=int, default=3,
        help="Maximum number of ref samples to draw per character/skeleton.",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Directory where ref.txt and tgt.txt are written. "
             "Defaults to <mode>/<category_lower>/ relative to this script.",
    )
    parser.add_argument(
        '--min_length', type=int, default=0,
        help="Minimum animation length in frames. Samples shorter than this are excluded. "
             "Requires --dataset_dir when non-zero.",
    )
    parser.add_argument(
        '--dataset_dir', type=str, default=None,
        help="Path to the directory containing .npy motion files. "
             "Required when --min_length > 0.",
    )
    parser.add_argument(
        '--no_self_pairs', action='store_true',
        help="Strictly forbid ref==tgt pairs. By default, self-pairs are used only "
             "as a last resort when no other tgt candidate is available.",
    )
    parser.add_argument(
        '--exclude_idle', action='store_true',
        help="Exclude any sample that appears in idle.txt from both ref and tgt pools.",
    )
    args = parser.parse_args()

    if args.min_length > 0 and args.dataset_dir is None:
        parser.error("--dataset_dir is required when --min_length > 0.")

    ref_cat, tgt_cat = parse_category(args.category)
    rng = random.Random(args.seed)

    ref_samples = load_category(ref_cat)
    tgt_samples = load_category(tgt_cat)

    if args.min_length > 0:
        print(f"Filtering by min_length={args.min_length} frames ...")
        ref_samples = filter_by_length(ref_samples, args.dataset_dir, args.min_length)
        tgt_samples = filter_by_length(tgt_samples, args.dataset_dir, args.min_length)

    if args.exclude_idle:
        idle_path = os.path.join(CATEGORY_DIR, 'idle.txt')
        if not os.path.exists(idle_path):
            print("  [exclude_idle] WARNING: idle.txt not found, skipping exclusion.")
        else:
            with open(idle_path) as f:
                idle_set = {line.strip() for line in f if line.strip()}
            before = len(ref_samples) + len(tgt_samples)
            ref_samples = [s for s in ref_samples if s not in idle_set]
            tgt_samples = [s for s in tgt_samples if s not in idle_set]
            removed = before - len(ref_samples) - len(tgt_samples)
            if removed:
                print(f"  [exclude_idle] Removed {removed} idle samples from pools.")

    if args.mode == 'in_skel':
        refs, tgts = sample_in_skel(ref_samples, tgt_samples, args.n_per_skel, rng, no_self_pairs=args.no_self_pairs)
    else:
        refs, tgts = sample_x_skel(ref_samples, tgt_samples, args.n_per_skel, rng)

    out_dir = args.output_dir or os.path.join(
        CATEGORY_DIR, args.mode, args.category.lower()
    )
    os.makedirs(out_dir, exist_ok=True)

    ref_path = os.path.join(out_dir, 'ref.txt')
    tgt_path = os.path.join(out_dir, 'tgt.txt')

    with open(ref_path, 'w') as f:
        f.write('\n'.join(refs) + '\n')
    with open(tgt_path, 'w') as f:
        f.write('\n'.join(tgts) + '\n')

    print(f"Generated {len(refs)} pairs  [{args.mode} | {args.category} | seed={args.seed}]")
    print(f"  ref → {ref_path}")
    print(f"  tgt → {tgt_path}")


if __name__ == '__main__':
    main()
    # python3 sample_blend_bench.py --mode in_skel --category Walk_Run --n_per_skel 20 --min_length 40 --dataset_dir ../../../dataset/truebones/zoo/truebones_processed/motions/