"""
Evaluation script for Mixamo retargeting outputs from sample/mix.py
(--control_mode tgt).

For each .json in the generation directory the script:
  1. Loads the generated BVH (data) and the ref_path npy (GT).
  2. Aligns both to the subset of joints defined by EVAL_ISOMO or EVAL_HOMEO,
     depending on the character group.
  3. Computes normalised MSE in global-position space.

Usage
-----
python -m eval.eval_mixamo_retargeting \
    --eval_gen_dir  <path/to/mix_output_dir> \
    [--gt_bvh_dir   <path/to/bvhs>]          \
    [--unique_str   _myrun]
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import Animation
import BVH
import numpy as np

from eval.utils.get_height import get_height

# ---------------------------------------------------------------------------
# Joint subsets
# ---------------------------------------------------------------------------

EVAL_ISOMO = [
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
]  # 22 joints

EVAL_HOMEO = [
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand',
]  # 25 joints

TEST_CHARS_ISOMO = ['Aj', 'BigVegas', 'Kaya', 'SportyGranny']
TEST_CHARS_HOMEO = ['Goblin_m', 'Mousey_m', 'Mremireh_m', 'Vampire_m']

THRESH = 50

def _eval_joints_for_char(char):
    if char in TEST_CHARS_ISOMO:
        return EVAL_ISOMO
    if char in TEST_CHARS_HOMEO:
        return EVAL_HOMEO
    return None


# ---------------------------------------------------------------------------
# Height computation
# ---------------------------------------------------------------------------

def compute_height(ref_bvh_path):
    """
    Character height via get_height(): sum of left-foot and right-foot bone
    chain lengths in the simplified skeleton, read from a GT BVH.
    """
    anim, joint_names, _ = BVH.load(ref_bvh_path)
    height = get_height(anim, joint_names)
    print(f'  height={height:.4f}  (from {os.path.basename(ref_bvh_path)})')
    return height


# ---------------------------------------------------------------------------
# Motion loaders
# ---------------------------------------------------------------------------

def load_bvh_positions(bvh_path, joint_subset):
    """
    Load a BVH and return global positions for the requested joint subset.

    Joint names are stripped of any namespace prefix (e.g. 'mixamorig:Hips' -> 'Hips').

    Returns
    -------
    pos : np.ndarray [T, len(joint_subset), 3]  — subset joints, same order as joint_subset list
    missing : list[str]  — joint names requested but absent in the BVH
    """
    anim, joint_names, _ = BVH.load(bvh_path)
    pos_all = Animation.positions_global(anim)   # [T, n_joints, 3]
    name_to_idx = {
        (n[n.find(':') + 1:] if ':' in n else n): i
        for i, n in enumerate(joint_names)
    }

    out_cols, missing = [], []
    for jname in joint_subset:
        if jname in name_to_idx:
            out_cols.append(pos_all[:, name_to_idx[jname], :])
        else:
            missing.append(jname)
    assert missing == [] or out_cols == [], 'Expected either all joints found or all missing'

    if not out_cols:
        return None, missing
    return np.stack(out_cols, axis=1), missing   # [T, J, 3]


def build_ambiguous_actions(motions_dir):
    """
    Scan motions_dir and return a set of (char, base_action_name) pairs where
    base_action_name is the action name *without* the (N) suffix, for every
    action that has at least one (N) variant on disk.

    E.g. if 'Kaya_Running (1)_42.npy' exists, ('Kaya', 'Running') is in the set,
    meaning both 'Running' and 'Running (1)' are ambiguous for Kaya.
    """
    ambiguous = set()
    for fname in os.listdir(motions_dir):
        if not fname.endswith('.npy'):
            continue
        stem = fname[:-4]
        m = re.search(r'\((\d+)\)', stem)
        if not m:
            continue
        # strip the (N) to get the base action name
        base_stem = stem[:m.start()].rstrip() + stem[m.end():]
        # base_stem is now e.g. 'Kaya_Running_42' — extract char and action
        parts = base_stem.split('_', 1)
        if len(parts) < 2:
            continue
        char = parts[0]
        action = re.sub(r'_\d+$', '', parts[1])   # strip trailing counter
        ambiguous.add((char, action))
    return ambiguous


def _is_ambiguous(npy_path, ambiguous_actions):
    """True if the npy stem's action name is in the ambiguous set."""
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    parts = stem.split('_', 1)
    if len(parts) < 2:
        return False
    char = parts[0]
    action = re.sub(r'\(\d+\)\s*', '', parts[1])  # strip any (N) to get base
    action = re.sub(r'_\d+$', '', action).strip()
    return (char, action) in ambiguous_actions


def npy_path_to_bvh(npy_path, gt_bvh_dir=None):
    """
    Resolve the GT BVH path for a given .npy path.

    If gt_bvh_dir is provided, use it directly; otherwise assume the BVH lives
    in a sibling 'bvhs/' directory next to 'motions/'.
    """
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    if gt_bvh_dir:
        return os.path.join(gt_bvh_dir, stem + '.bvh')
    motions_dir = os.path.dirname(os.path.abspath(npy_path))
    parent = os.path.dirname(motions_dir)
    bvhs_dir = os.path.join(parent, 'bvhs')
    return os.path.join(bvhs_dir, stem + '.bvh')


# ---------------------------------------------------------------------------
# MSE metric
# ---------------------------------------------------------------------------

def normalised_mse(pos, pos_ref, height):
    """
    Per-joint, per-frame squared error summed over XYZ, normalised by height,
    then averaged over joints and frames.

    Matches the reference:
        err = 1. / ref_height * ((pos_ref - pos) ** 2).sum(axis=-1)  # [T, J]
    """
    err = (1. / height) * ((pos_ref - pos) ** 2).sum(axis=-1)  # [T, J]
    return float(err.mean())


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(gen_bvh_path, ref_bvh_path, char, height_cache, ref_s=None, ref_e=None):
    """
    Compute normalised MSE between the generated BVH and the GT BVH for one
    sample, restricted to the canonical joint subset for *char*.

    ref_s / ref_e : temporal window into the GT BVH that the generation was
                    conditioned on (from JSON regions.ref_s / regions.ref_e).
                    When provided, ref_pos is sliced to [ref_s:ref_e] so the
                    comparison uses the correct GT frames rather than the clip
                    start.

    Returns a float MSE or None on failure.
    """
    eval_joints = _eval_joints_for_char(char)
    if eval_joints is None:
        print(f'  WARNING: unknown character [{char}], skipping.')
        return None

    # height (cached per character; derived from the GT BVH)
    if char not in height_cache:
        try:
            height_cache[char] = compute_height(ref_bvh_path)
        except Exception as e:
            print(f'  WARNING: could not compute height for [{char}]: {e}')
            return None
    height = height_cache[char]

    # Load generated BVH positions
    try:
        gen_pos, missing_gen = load_bvh_positions(gen_bvh_path, eval_joints)
    except Exception as e:
        print(f'  WARNING: failed to load generated BVH [{gen_bvh_path}]: {e}')
        return None
    if gen_pos is None:
        print(f'  WARNING: no eval joints found in generated BVH [{gen_bvh_path}]')
        return None
    if missing_gen:
        print(f'  WARNING: joints missing in generated BVH: {missing_gen}')

    # Load GT BVH positions and slice to the conditioned window
    try:
        ref_pos, missing_ref = load_bvh_positions(ref_bvh_path, eval_joints)
    except Exception as e:
        print(f'  WARNING: failed to load GT BVH [{ref_bvh_path}]: {e}')
        return None
    if ref_pos is None:
        print(f'  WARNING: no eval joints found in GT BVH [{ref_bvh_path}]')
        return None
    if missing_ref:
        print(f'  WARNING: joints missing in GT BVH: {missing_ref}')

    if ref_s is not None and ref_e is not None:
        ref_pos = ref_pos[ref_s:ref_e]

    # Align temporal lengths (truncate to shorter)
    T = min(gen_pos.shape[0], ref_pos.shape[0])
    if T == 0:
        return None

    gen_pos = gen_pos[:T]
    ref_pos = ref_pos[:T]

    # Only evaluate joints present in both
    present_gen = set(eval_joints) - set(missing_gen)
    present_ref = set(eval_joints) - set(missing_ref)
    common = [j for j in eval_joints if j in present_gen and j in present_ref]
    if not common:
        print(f'  WARNING: no common joints for [{char}], skipping.')
        return None

    if len(common) < len(eval_joints):
        common_idx = [i for i, j in enumerate(eval_joints) if j in set(common)]
        gen_pos = gen_pos[:, common_idx, :]
        ref_pos = ref_pos[:, common_idx, :]

    # Root-relative: subtract Hips by name lookup in the surviving joint list
    if 'Hips' not in common:
        print(f'  WARNING: Hips not in common joints for [{char}], skipping.')
        return None
    
    hips_idx = common.index('Hips')
    gen_pos = gen_pos - gen_pos[:, hips_idx:hips_idx + 1, :]
    ref_pos = ref_pos - ref_pos[:, hips_idx:hips_idx + 1, :]

    mse = normalised_mse(gen_pos, ref_pos, height)
    print(f'  MSE={mse:.4f}  (over {len(common)} joints, {T} frames)')

    return mse


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_eval(args):
    gen_dir = args.eval_gen_dir

    json_files = sorted(f for f in os.listdir(gen_dir) if f.endswith('.json'))
    print(f'Found {len(json_files)} JSON files in [{gen_dir}]')

    # Build ambiguous-action set once from the motions directory
    ambiguous_actions = set()
    if json_files:
        with open(os.path.join(gen_dir, json_files[0])) as _f:
            _meta = json.load(_f)
        _motions_dir = os.path.dirname(os.path.abspath(_meta['samples']['ref_path']))
        if os.path.isdir(_motions_dir):
            ambiguous_actions = build_ambiguous_actions(_motions_dir)
            print(f'  {len(ambiguous_actions)} ambiguous action names found in [{_motions_dir}]')

    height_cache = {}
    results_by_char = defaultdict(list)   # char -> list[float MSE]
    results_iso = []
    results_homeo = []
    results_all = []
    per_sample = []   # list of {'stem', 'gen_bvh', 'ref_bvh', 'char', 'mse'}
    trash_bin = 0
    duplicate_bin = 0

    for json_fname in json_files:
        json_path = os.path.join(gen_dir, json_fname)
        with open(json_path) as f:
            meta = json.load(f)

        stem = os.path.splitext(json_fname)[0]
        gen_bvh = os.path.join(gen_dir, stem + '.bvh')
        ref_bvh = npy_path_to_bvh(meta['samples']['ref_path'], args.gt_bvh_dir)

        # Character is the ref object (the skeleton we generated on)
        char = meta['samples']['ref_object']

        if not os.path.isfile(gen_bvh):
            print(f'  WARNING: generated BVH not found [{gen_bvh}], skipping.')
            continue
        if not os.path.isfile(ref_bvh):
            print(f'  WARNING: GT BVH not found [{ref_bvh}], skipping.')
            continue

        # Skip samples where ref or tgt action has duplicate takes in the dataset
        # (either the stem itself has (N), or a (N) variant exists for its action)
        _ref_path = meta['samples']['ref_path']
        _tgt_path = meta['samples']['tgt_path']
        if (_is_ambiguous(_ref_path, ambiguous_actions) or
                _is_ambiguous(_tgt_path, ambiguous_actions)):
            print(f'  [ambiguous] skipping [{json_fname}]')
            duplicate_bin += 1
            continue

        ref_s = meta.get('regions', {}).get('ref_s')
        ref_e = meta.get('regions', {}).get('ref_e')

        print(f'  [{json_fname}] char={char}  ref_window=[{ref_s}:{ref_e}]')
        mse = evaluate_sample(gen_bvh, ref_bvh, char, height_cache, ref_s=ref_s, ref_e=ref_e)
    
        if mse is None:
            continue

        if mse > THRESH:
            trash_bin += 1
            continue

        per_sample.append({'stem': stem, 'gen_bvh': gen_bvh, 'ref_bvh': ref_bvh, 'char': char, 'mse': mse})
        results_by_char[char].append(mse)
        results_all.append(mse)
        if char in TEST_CHARS_ISOMO:
            results_iso.append(mse)
        elif char in TEST_CHARS_HOMEO:
            results_homeo.append(mse)

    # ---------------------------------------------------------------------------
    # Aggregate & print
    # ---------------------------------------------------------------------------
    def _agg(vals, label):
        if not vals:
            print(f'  [{label}] no results')
            return {}
        arr = np.array(vals, dtype=np.float64)
        d = {
            'mean':   float(np.mean(arr)),
            'std':    float(np.std(arr)),
            'median': float(np.median(arr)),
            'n':      len(arr),
        }
        print(f'  [{label}] MSE mean={d["mean"]:.4f} ± {d["std"]:.4f}  '
              f'median={d["median"]:.4f}  (n={d["n"]})')
        return d

    print(f'\nDiscarded {duplicate_bin} duplicate-take samples (ref or tgt stem contains (N))')
    print(f'Discarded {trash_bin} samples with MSE > {THRESH:.4f} (considered failed retargeting)')

    print('\n===== RESULTS =====')
    out = {}
    out['ALL']   = _agg(results_all,  'ALL')
    out['ISOMO'] = _agg(results_iso,  'ISOMO')
    out['HOMEO'] = _agg(results_homeo,'HOMEO')

    print('\n  --- Per-character ---')
    out['per_char'] = {}
    for char in sorted(results_by_char):
        out['per_char'][char] = _agg(results_by_char[char], char)

    # Worst-N samples
    n_worst = args.n_worst
    worst = sorted(per_sample, key=lambda x: x['mse'], reverse=True)[:n_worst]
    print(f'\n  --- Worst {n_worst} samples ---')
    for rank, s in enumerate(worst, 1):
        print(f'  #{rank:2d}  MSE={s["mse"]:.4f}  char={s["char"]}  {s["stem"]}')
        print(f'        gen: {s["gen_bvh"]}')
        print(f'        ref: {s["ref_bvh"]}')
    out['worst'] = [{'rank': i + 1, **s} for i, s in enumerate(worst)]

    # Save JSON
    bench_name = os.path.basename(os.path.normpath(gen_dir))
    out_json = os.path.join(
        os.path.dirname(os.path.normpath(gen_dir)),
        bench_name + f'__retargeting_mse{args.unique_str}.json',
    )

    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with open(out_json, 'w') as fw:
        json.dump(_safe(out), fw, indent=2)
    print(f'\nSaved results to [{out_json}]')
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Mixamo retargeting outputs (MSE vs GT).')
    parser.add_argument('--eval_gen_dir', required=True,
                        help='Directory with generated BVH + JSON files from mix.py.')
    parser.add_argument('--gt_bvh_dir', default=None,
                        help='Directory containing GT BVH files. If omitted, resolved '
                             'automatically as a sibling bvhs/ dir next to the motions/ dir '
                             'referenced in each JSON.')
    parser.add_argument('--n_worst', default=5, type=int,
                        help='Number of worst samples to report (default: 5).')
    parser.add_argument('--unique_str', default='',
                        help='Suffix appended to the output JSON filename.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    assert os.path.isdir(args.eval_gen_dir), f'eval_gen_dir not found: {args.eval_gen_dir}'
    run_eval(args)
