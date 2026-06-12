"""
Evaluation script for motion blending outputs from sample/mix.py.

Evaluates each generated blend against its source (ref, tgt) GT motions and the
ref character's full GT pool. Mirrors the AnyTop eval metrics where applicable.

Usage:
    python -m eval.eval_truebones_blend \
        --eval_gen_dir <path/to/mix_output> \
        --eval_gt_dir  <path/to/gt_bvh_or_npy_dir> \
        --source       npy       # or bvh
        --feats        rot       # or loc (FK-derived positions for bvh, stored positions for npy)

Fidelity metrics  (per-triple — lower is better)
-------------------------------------------------
pre_fidelity    : patched-NN distance between blend[ref_s:overlap_s] (pure-ref zone) and the
                  full character GT pool.  "Does this zone look like something this character does?"
post_fidelity   : same for blend[overlap_e:tgt_e] (pure-tgt zone, same-skeleton only).
trans_fidelity  : same for blend[overlap_s:overlap_e] (transition zone).  Key naturalness indicator.

Diversity metrics  (per-triple + group-level)
---------------------------------------------
local_diversity     : per-window NN distance from the full blend to the concatenated GT pool.
                      Dense local fidelity to the character's distribution at every moment.
local_diversity_trans : same as local_diversity but restricted to the transition zone only.
intra_diversity     : avg distance between random window pairs within the blend.
                      "How internally varied is this blend?"
intra_div_gt_diff   : |intra_diversity - gt_intra_diversity|.
                      How much the blend's internal variety deviates from the character's GT baseline.
intra_diversity_trans    : intra_diversity restricted to the transition zone only.
intra_div_gt_diff_trans  : |intra_diversity_trans - gt_intra_diversity|. Transition-specific naturalness.
inter_diversity     : [group-level] avg distance between repetitions of the same (ref, tgt, alpha) pair.
gt_intra_diversity  : [group-level] character-level mean intra-diversity over the GT pool (baseline).
coverage            : [group-level] fraction of the ref character's full GT distribution covered by any
                      blend for that character (threshold = 20° / 0.1 m).

Smoothness metrics  (per-triple — lower is better, ratio near 1 is ideal)
--------------------------------------------------------------------------
global_jerk         : mean jerk norm over the full blend.
pre_jerk            : jerk in the pure-ref zone (before transition).
trans_jerk          : jerk in the overlap / transition zone (key smoothness indicator).
post_jerk           : jerk in the pure-tgt zone (after transition).
ref_gt_jerk         : GT ref jerk cropped to the pure-ref window (baseline for pre_jerk).
tgt_gt_jerk         : GT tgt jerk cropped to the pure-tgt window (baseline for post_jerk).
ref_trans_gt_jerk   : GT ref jerk at the transition-corresponding window.
tgt_trans_gt_jerk   : GT tgt jerk at the transition-corresponding window (tgt start).
jerk_ratio          : trans_jerk / gt_pool_jerk.  > 1 = jerkier than the character's typical motion.
"""

import json as json_module
import math
import os
from collections import defaultdict

import Animation  # pip install git+https://github.com/inbar-2344/Motion.git
import BVH        # pip install git+https://github.com/inbar-2344/Motion.git
from utils.bvh_io import bvh_load_safe
import numpy as np
import torch

from eval.metrics.distances import avg_per_frame_dist
from eval.metrics.patched_nn import patched_nn_main
from eval.metrics.perwindow_nn import coverage, perwindow_nn
from utils.fixseed import fixseed
from utils.parser_util import evaluation_parser

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def bvh2data(bvh_file_path, feats='rot', root_relative=False):
    """BVH -> flattened feature tensor [n_frames, n_joints * d].

    Joints are sorted alphabetically so that blend outputs and GT files are
    always comparable by position, regardless of BVH hierarchy traversal order
    (which Blender may change on export).

    feats='rot' : 6D rotation representation  [n_frames, n_joints * 6]
    feats='loc' : world joint positions via FK [n_frames, n_joints * 3]

    root_relative (loc only): subtract root position from all joints and
        exclude the root joint (always zero after subtraction).
        Makes the representation trajectory-independent.
    """
    anim, joint_names, _ = bvh_load_safe(bvh_file_path)
    sorted_idx = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
    if feats == 'rot':
        r = torch.from_numpy(anim.rotations.rotation_matrix(cont6d=True))
        r = r[:, sorted_idx, :]   # [n_frames, n_joints, 6]
        return r.view(r.shape[0], -1)
    elif feats == 'loc':
        pos = Animation.positions_global(anim)   # [n_frames, n_joints, 3]; root at [:, 0, :]
        if root_relative:
            pos = pos - pos[:, 0:1, :]           # subtract root; root → (0,0,0)
            idx = [i for i in sorted_idx if i != 0]  # drop root (always zero)
        else:
            idx = sorted_idx
        pos = torch.from_numpy(pos[:, idx, :])
        return pos.reshape(pos.shape[0], -1).double()
    raise ValueError(f'Unknown feats [{feats}]')


def npy2data(npy_file_path, feats):
    """Processed .npy -> [n_frames, features].  Assumes unnormalized data."""
    try:
        anim = np.load(npy_file_path)
    except Exception:
        anim = np.load(npy_file_path, allow_pickle=True).item()
        anim = anim['motion_raw'].transpose(0, 3, 1, 2)
    y_root = anim[..., 0, 1]
    anim = anim[..., 1:, :]
    if feats == 'rot':
        anim_out = torch.from_numpy(anim[..., 3:9]).reshape(anim.shape[:-2] + (-1,))
    elif feats == 'loc':
        anim_out = torch.from_numpy(anim[..., :3])
        anim_out[..., 1] = anim_out[..., 1] - y_root[..., np.newaxis]
        anim_out = anim_out.reshape(anim.shape[:-2] + (-1,))
    else:
        raise ValueError(f'Unknown feats [{feats}]')
    if anim_out.dtype == torch.float32:
        anim_out = anim_out.double()
    return anim_out


def load_motion(path, source, feats, root_relative=False):
    """Load a motion file using the specified source format and feature type."""
    if source == 'bvh':
        return bvh2data(path, feats, root_relative=root_relative)
    elif source == 'npy':
        return npy2data(path, feats)
    raise ValueError(f'Unknown source [{source}]')


def source_ext(source):
    """Return the file extension (with dot) for a given source format."""
    return '.bvh' if source == 'bvh' else '.npy'


def resolve_blend_path(json_file, gen_dir, source):
    """Find the blend output file for a given JSON using the source extension.

    Ignores meta['output'] — --source is authoritative for file format.
    """
    stem = os.path.splitext(json_file)[0]
    return os.path.join(gen_dir, stem + source_ext(source))


def resolve_gt_path(meta_path, gt_dir, source):
    """Resolve the GT file in gt_dir for a source path from the JSON metadata.

    The meta_path may be in the wrong format/directory; we map only the basename
    (without extension) into gt_dir with the extension implied by --source.
    """
    basename = os.path.splitext(os.path.basename(meta_path))[0]
    return os.path.join(gt_dir, basename + source_ext(source))


def gt_dir_for_source(base_dir, source):
    """Resolve the GT directory for a given source format.

    Accepts either:
    - A fully-qualified path already ending in the correct subfolder — used as-is.
    - A fully-qualified path ending in the *wrong* subfolder — swapped to sibling.
    - A base dataset directory — the correct subfolder is appended.
    """
    subfolder = 'bvhs' if source == 'bvh' else 'motions'
    norm = os.path.normpath(base_dir)
    basename = os.path.basename(norm)
    if basename == subfolder:
        return norm
    if basename in ('bvhs', 'motions'):
        return os.path.join(os.path.dirname(norm), subfolder)
    return os.path.join(norm, subfolder)


def character2data(dir_path, character_name, source, feats, root_relative=False):
    """Load all motion files for a given character from dir_path."""
    ext = source_ext(source)
    files = sorted([
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.endswith(ext) and f.split('_')[0] == character_name
    ])
    if not files:
        return None
    data = []
    for path in files:
        sample = load_motion(path, source, feats, root_relative=root_relative)
        if sample.shape[0] > 20:
            data.append(sample)
    print(f'[{character_name}] GT pool: {len(data)} clips')
    return data if data else None


# ---------------------------------------------------------------------------
# Leaf-joint mask
# ---------------------------------------------------------------------------

def get_leaf_mask(bvh_file_path, feats, root_relative=False):
    """Return a per-joint boolean numpy array (True = leaf joint) aligned with
    the feature columns produced by bvh2data for the same feats/root_relative.

    A leaf joint is one that has no children in the BVH hierarchy.
    Only meaningful for source='bvh'; returns None for other sources.
    """
    anim, joint_names, _ = bvh_load_safe(bvh_file_path)
    n = len(joint_names)
    is_leaf = np.ones(n, dtype=bool)
    for p in anim.parents:
        if p >= 0:
            is_leaf[p] = False
    sorted_idx = sorted(range(n), key=lambda i: joint_names[i])
    if feats == 'loc' and root_relative:
        idx = [i for i in sorted_idx if i != 0]
    else:
        idx = sorted_idx
    return np.array([is_leaf[i] for i in idx])   # [n_feature_joints]


# ---------------------------------------------------------------------------
# Jerk metric
# ---------------------------------------------------------------------------

def rotation_jerk(motion_rot):
    """
    Mean norm of the 3rd-order finite difference of the 6D rotation features.
    Used as a smoothness proxy (true positional jerk would need FK).
    Returns NaN if fewer than 4 frames.
    """
    if motion_rot.shape[0] < 4:
        return float('nan')
    jerk = torch.diff(motion_rot.double(), n=3, dim=0)  # [T-3, features]
    return jerk.norm(dim=-1).mean().item()


def rotation_jerk_subset(motion, joint_mask, feat_dim):
    """Jerk restricted to joints where joint_mask is True.

    joint_mask : bool array [n_joints]
    feat_dim   : features per joint (3 for loc, 6 for rot)
    """
    if joint_mask is None or not joint_mask.any():
        return float('nan')
    col_mask = np.repeat(joint_mask, feat_dim)
    return rotation_jerk(motion[:, col_mask])



# ---------------------------------------------------------------------------
# Per-triple evaluation
# ---------------------------------------------------------------------------

def evaluate_blend_triple(blend_rot, meta, ref_rot, tgt_rot, source, feats,
                          gt_pool=None, gt_pool_jerk=None, gt_pool_jerk_structural=None,
                          gt_pool_jerk_leaf=None, leaf_mask=None,
                          gt_intra_diversity=float('nan'),
                          tmin=15, n_intra_reps=10):
    """
    Compute all per-triple metrics.

    Parameters
    ----------
    blend_rot  : [T, features] tensor — the generated blend
    meta       : JSON dict with 'regions' and 'samples' keys
    ref_rot    : [T_ref, features] GT ref motion (used for jerk baselines only)
    tgt_rot    : [T_tgt, features] GT tgt motion, or None (used for jerk baselines only)
    source     : 'bvh' or 'npy'
    feats      : 'rot' or 'loc'
    gt_pool    : list of [T_i, features] tensors — full character GT pool for pre/post/trans_nn
    """
    use_pos = feats == 'loc'
    T         = blend_rot.shape[0]

    r         = meta['regions']
    ref_s     = r['ref_s']
    ref_e     = min(r['ref_e'],     T)
    tgt_s     = r['tgt_s']
    tgt_e     = min(r['tgt_e'],     T)
    overlap_s = r['overlap_s']
    overlap_e = min(r['overlap_e'], T)
    same_skel = r['same_skeleton']

    pure_ref_len = overlap_s - ref_s
    trans_len    = overlap_e - overlap_s
    pure_tgt_len = tgt_e - overlap_e

    # -- Zone NN metrics vs full GT pool --------------------------------------
    # pre_fidelity / post_fidelity / trans_fidelity: "does this zone look like something this character does?"
    # Database is the full character GT pool (same question as coverage, partitioned by zone).
    # gt_pool is pre-augmented by the caller to include all source clips for this character.
    _have_pool = gt_pool is not None and len(gt_pool) > 0
    def _pool_nn(query):
        compatible = [c for c in gt_pool if c.shape[-1] == query.shape[-1]]
        if not compatible:
            return float('nan')
        val = min(
            patched_nn_main(query, clip, tmin=tmin, use_pos=use_pos)
            for clip in compatible
        )
        return float('nan') if not np.isfinite(val) else val
    pre_nn = (
        _pool_nn(blend_rot[ref_s:overlap_s])
        if (_have_pool and pure_ref_len >= tmin) else float('nan')
    )
    post_nn = (
        _pool_nn(blend_rot[overlap_e:tgt_e])
        if (_have_pool and same_skel and pure_tgt_len >= tmin) else float('nan')
    )
    trans_nn = (
        _pool_nn(blend_rot[overlap_s:overlap_e])
        if (_have_pool and trans_len >= tmin) else float('nan')
    )

    # -- Diversity metrics -----------------------------------------------------
    # local_diversity: per-window NN to the full GT pool (dense, no partition constraint).
    _compat_pool = [c for c in gt_pool if c.shape[-1] == blend_rot.shape[-1]] if _have_pool else []
    gt_pool_cat = torch.cat(_compat_pool, dim=0) if _compat_pool else None
    local_diversity = (
        perwindow_nn(blend_rot, gt_pool_cat, tmin=tmin, use_pos=use_pos)
        if gt_pool_cat is not None else float('nan')
    )

    # local_diversity_trans: same as local_diversity but only over the transition zone.
    local_diversity_trans = (
        perwindow_nn(blend_rot[overlap_s:overlap_e], gt_pool_cat, tmin=tmin, use_pos=use_pos)
        if (gt_pool_cat is not None and trans_len >= tmin) else float('nan')
    )

    # intra_diversity: avg distance between random window pairs within this blend.
    if T > 2 * tmin:
        offsets   = np.random.randint(T - tmin, size=(n_intra_reps, 2))
        intra_vals = [
            avg_per_frame_dist(
                blend_rot[o[0]:o[0]+tmin],
                blend_rot[o[1]:o[1]+tmin],
                norm=feats,
            )
            for o in offsets
        ]
        intra_diversity = float(np.mean(intra_vals))
    else:
        intra_diversity = float('nan')

    # intra_div_gt_diff: how much this blend's internal variety deviates from the GT baseline.
    intra_div_gt_diff = (
        abs(intra_diversity - gt_intra_diversity)
        if (not math.isnan(intra_diversity) and not math.isnan(gt_intra_diversity))
        else float('nan')
    )

    # intra_diversity_trans: same as intra_diversity but windows sampled only from the transition zone.
    trans_zone = blend_rot[overlap_s:overlap_e]
    T_trans = trans_zone.shape[0]
    if T_trans >= 2 * tmin:
        offsets_trans = np.random.randint(T_trans - tmin, size=(n_intra_reps, 2))
        intra_vals_trans = [
            avg_per_frame_dist(
                trans_zone[o[0]:o[0]+tmin],
                trans_zone[o[1]:o[1]+tmin],
                norm=feats,
            )
            for o in offsets_trans
        ]
        intra_diversity_trans = float(np.mean(intra_vals_trans))
    else:
        intra_diversity_trans = float('nan')

    intra_div_gt_diff_trans = (
        abs(intra_diversity_trans - gt_intra_diversity)
        if (not math.isnan(intra_diversity_trans) and not math.isnan(gt_intra_diversity))
        else float('nan')
    )

    # -- Jerk (3rd-order diff in rot/loc feature space) ----------------------
    # GT baselines are cropped to the same temporal windows as the blend zones
    # so that ref_gt_jerk / tgt_gt_jerk remain valid per-sample diagnostics.
    # jerk_ratio uses gt_pool_jerk (character-level mean) as its baseline.
    global_jerk = rotation_jerk(blend_rot)
    pre_jerk    = rotation_jerk(blend_rot[ref_s    : overlap_s]) if pure_ref_len >= 4 else float('nan')
    trans_jerk  = rotation_jerk(blend_rot[overlap_s: overlap_e]) if trans_len    >= 4 else float('nan')
    post_jerk   = rotation_jerk(blend_rot[overlap_e: tgt_e])     if pure_tgt_len >= 4 else float('nan')

    # GT jerk in the window that corresponds to each blend zone
    ref_gt_jerk = (
        rotation_jerk(ref_rot[: pure_ref_len])
        if pure_ref_len >= 4 else float('nan')
    )
    tgt_gt_jerk = (
        rotation_jerk(tgt_rot[trans_len : trans_len + pure_tgt_len])
        if (tgt_rot is not None and pure_tgt_len >= 4) else float('nan')
    )
    ref_trans_gt_jerk = (
        rotation_jerk(ref_rot[pure_ref_len : pure_ref_len + trans_len])
        if trans_len >= 4 else float('nan')
    )
    tgt_trans_gt_jerk = (
        rotation_jerk(tgt_rot[: trans_len])
        if (tgt_rot is not None and trans_len >= 4) else float('nan')
    )
    # jerk_ratio: transition jerk relative to the character's typical jerk level.
    # Baseline is the mean jerk over the full GT pool (same character-level
    # distribution used for coverage/pre_fidelity/post_fidelity/trans_fidelity), so the ratio is
    # not biased by the specific source clips chosen for this blend.
    jerk_baseline = gt_pool_jerk if (gt_pool_jerk is not None and not math.isnan(gt_pool_jerk)) else float('nan')
    jerk_ratio     = (
        trans_jerk / jerk_baseline
        if (not math.isnan(trans_jerk) and not math.isnan(jerk_baseline) and jerk_baseline > 0)
        else float('nan')
    )

    # Structural vs leaf split — only for trans zone (the diagnostic of interest).
    feat_dim = 3 if feats == 'loc' else 6
    n_joints_blend = blend_rot.shape[-1] // feat_dim
    _leaf_mask_ok = (
        leaf_mask is not None and len(leaf_mask) == n_joints_blend
    )
    if leaf_mask is not None and not _leaf_mask_ok:
        print(f'  WARNING: leaf_mask size {len(leaf_mask)} does not match blend joints '
              f'{n_joints_blend} — skipping structural/leaf jerk split')
    _structural_mask = (~leaf_mask) if _leaf_mask_ok else None
    _leaf_mask       = leaf_mask if _leaf_mask_ok else None

    trans_jerk_structural = (
        rotation_jerk_subset(blend_rot[overlap_s:overlap_e], _structural_mask, feat_dim)
        if trans_len >= 4 else float('nan')
    )
    trans_jerk_leaf = (
        rotation_jerk_subset(blend_rot[overlap_s:overlap_e], _leaf_mask, feat_dim)
        if trans_len >= 4 else float('nan')
    )

    def _ratio(tj, baseline):
        b = baseline if (baseline is not None and not math.isnan(baseline) and baseline > 0) else float('nan')
        return tj / b if (not math.isnan(tj) and not math.isnan(b)) else float('nan')

    jerk_ratio_structural = _ratio(trans_jerk_structural, gt_pool_jerk_structural)
    jerk_ratio_leaf       = _ratio(trans_jerk_leaf,       gt_pool_jerk_leaf)

    return {
        # -- Fidelity: how natural is each zone vs the full character GT distribution --
        'pre_fidelity':   pre_nn,
        'post_fidelity':  post_nn,
        'trans_fidelity': trans_nn,
        # -- Diversity: variety within the blend and deviation from GT baseline --------
        'local_diversity':        local_diversity,
        'local_diversity_trans':  local_diversity_trans,
        'intra_diversity':        intra_diversity,
        'intra_div_gt_diff':      intra_div_gt_diff,
        'intra_diversity_trans':  intra_diversity_trans,
        'intra_div_gt_diff_trans': intra_div_gt_diff_trans,
        # -- Smoothness: jerk in feature space ----------------------------------------
        'global_jerk':        global_jerk,
        'pre_jerk':           pre_jerk,
        'trans_jerk':         trans_jerk,
        'post_jerk':          post_jerk,
        'ref_gt_jerk':        ref_gt_jerk,
        'tgt_gt_jerk':        tgt_gt_jerk,
        'ref_trans_gt_jerk':  ref_trans_gt_jerk,
        'tgt_trans_gt_jerk':  tgt_trans_gt_jerk,
        'jerk_ratio':              jerk_ratio,
        'trans_jerk_structural':   trans_jerk_structural,
        'trans_jerk_leaf':         trans_jerk_leaf,
        'jerk_ratio_structural':   jerk_ratio_structural,
        'jerk_ratio_leaf':         jerk_ratio_leaf,
        # -- Metadata -----------------------------------------------------------------
        'same_skeleton':  same_skel,
    }


# ---------------------------------------------------------------------------
# Aggregation & printing
# ---------------------------------------------------------------------------

def aggregate_results(results):
    _skip = {'same_skeleton', 'sample'}
    scalar_keys = [k for k in results[0] if k not in _skip]
    out = {}
    for m in scalar_keys:
        vals = []
        for r in results:
            v = r[m]
            try:
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
        if vals:
            arr = np.array(vals, dtype=np.float64)
            q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            out[m] = {
                'mean':   float(np.mean(arr)),
                'std':    float(np.std(arr)),
                'median': float(np.median(arr)),
                'iqr':    q3 - q1,
                'n':      len(vals),
            }
        else:
            out[m] = {'mean': float('nan'), 'std': float('nan'),
                      'median': float('nan'), 'iqr': float('nan'), 'n': 0}
    return out


_JERK_KEYS = {
    'global_jerk', 'pre_jerk', 'trans_jerk', 'post_jerk',
    'ref_gt_jerk', 'tgt_gt_jerk', 'ref_trans_gt_jerk', 'tgt_trans_gt_jerk',
    'jerk_ratio',
    'trans_jerk_structural', 'trans_jerk_leaf',
    'jerk_ratio_structural', 'jerk_ratio_leaf',
}

def print_results(label, eval_dict):
    print(f'[{label}] RESULTS:')
    print('=' * 10)
    for metric, vals in eval_dict.items():
        if math.isnan(vals['mean']):
            print(f'  [{metric}] N/A')
        elif metric in _JERK_KEYS:
            print(f'  [{metric}] mean={vals["mean"]:.4f}±{vals["std"]:.4f}  median={vals["median"]:.4f} IQR={vals["iqr"]:.4f}  (n={vals["n"]})')
        else:
            print(f'  [{metric}] {vals["mean"]:.4f} +/- {vals["std"]:.4f}  (n={vals["n"]})')
    print('=' * 10)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def eval_blend_benchmark(args):
    source        = args.source
    feats         = args.feats
    root_relative = args.root_relative
    use_pos = feats == 'loc'
    tmin               = 15
    coverage_threshold = 0.25 if feats == 'loc' else math.radians(20)
    # Adaptive coverage threshold: fraction of the character's gt_intra_diversity.
    # Makes coverage scale-invariant across characters of different sizes.
    # Falls back to the fixed threshold if gt_intra_diversity is unavailable.
    coverage_adaptive_alpha = 0.75
    MAX_GT_CLIPS       = 30                 # cap GT pool to avoid OOM (Trex: 76 clips)

    gt_dir = gt_dir_for_source(args.eval_gt_dir, source)
    print(f'GT directory: [{gt_dir}]')

    json_files = sorted(f for f in os.listdir(args.eval_gen_dir) if f.endswith('.json'))
    print(f'Found [{len(json_files)}] blend JSON files in [{args.eval_gen_dir}]')

    # ------- Load benchmark sample lists for pool prioritisation -------------
    # All paths listed in ref_samples / tgt_samples define the "relevant" GT clips
    # for this benchmark (e.g. all walk + run clips for a walk→run benchmark).
    # These are loaded per-character and used to fill the GT pool before random clips.
    def _read_sample_list(path):
        if not path or not os.path.exists(path):
            return []
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    _ref_lines = _read_sample_list(args.ref_samples)
    _tgt_lines = _read_sample_list(args.tgt_samples)
    # Build per-character set of relevant GT paths (resolved to gt_dir)
    char_relevant_paths = defaultdict(set)
    for line in _ref_lines + _tgt_lines:
        p = resolve_gt_path(line, gt_dir, source)
        # character name: first token before '___' in the basename
        basename = os.path.splitext(os.path.basename(p))[0]
        char_name = basename.split('___')[0] if '___' in basename else basename.split('_')[0]
        char_relevant_paths[char_name].add(p)

    # ------- Group JSON files by ref character --------------------------------
    char_groups = defaultdict(list)  # character -> [(json_file, meta), ...]
    n_self_pairs = 0
    for json_file in json_files:
        json_path = os.path.join(args.eval_gen_dir, json_file)
        meta      = json_module.load(open(json_path))
        if getattr(args, 'exclude_self_pairs', False):
            if meta['samples'].get('ref') == meta['samples'].get('tgt'):
                n_self_pairs += 1
                continue
        char      = meta['samples']['ref_object']
        char_groups[char].append((json_file, meta))
    if n_self_pairs:
        print(f'Excluded {n_self_pairs} self-pair(s) (ref == tgt) from evaluation.')

    # ------- Helpers ----------------------------------------------------------
    def _agg(vals):
        clean = []
        for v in vals:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    clean.append(fv)
            except (TypeError, ValueError):
                pass
        if not clean:
            return {'mean': float('nan'), 'std': float('nan'),
                    'median': float('nan'), 'iqr': float('nan'), 'n': 0}
        arr = np.array(clean, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        return {
            'mean':   float(np.mean(arr)),
            'std':    float(np.std(arr)),
            'median': float(np.median(arr)),
            'iqr':    q3 - q1,
            'n':      len(clean),
        }

    def _cov_dict(cov_vals):
        clean = []
        for v in cov_vals:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    clean.append(fv * 100)
            except (TypeError, ValueError):
                pass
        if not clean:
            return {'mean': float('nan'), 'std': float('nan'),
                    'median': float('nan'), 'iqr': float('nan'), 'n': 0}
        arr = np.array(clean, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        return {
            'mean':   float(np.mean(arr)),
            'std':    float(np.std(arr)),
            'median': float(np.median(arr)),
            'iqr':    q3 - q1,
            'n':      len(clean),
        }

    # ------- Accumulators across all characters ------------------------------
    all_results       = []   # list of metric dicts
    all_same_skel     = []   # bool per result (mirrors all_results order)
    inter_div_all      = []   # inter-diversity scalars (all pairs)
    inter_div_in       = []   # inter-diversity scalars (same-skeleton pairs only)
    coverage_all       = []   # per-character coverage scalars (all)
    coverage_in        = []   # per-character coverage scalars (same-skeleton subset)
    coverage_pre_all   = []
    coverage_pre_in    = []
    coverage_trans_all = []
    coverage_trans_in  = []
    coverage_post_all  = []
    coverage_post_in   = []
    gt_intra_div_all   = []   # per-character gt_intra_diversity scalars

    # ------- Per-character loop ----------------------------------------------
    for char, items in char_groups.items():
        gt_pool = character2data(gt_dir, char, source, feats, root_relative=root_relative)

        # Leaf-joint mask from the first BVH file for this character in gt_dir.
        leaf_mask = None
        if source == 'bvh':
            _bvh_candidates = sorted(
                f for f in os.listdir(gt_dir)
                if f.endswith('.bvh') and os.path.splitext(f)[0].split('_')[0] == char
            )
            if _bvh_candidates:
                leaf_mask = get_leaf_mask(
                    os.path.join(gt_dir, _bvh_candidates[0]), feats, root_relative
                )

        gt_pool_jerk = (
            float(np.nanmean([rotation_jerk(t) for t in gt_pool]))
            if gt_pool else float('nan')
        )
        feat_dim = 3 if feats == 'loc' else 6
        gt_pool_jerk_structural = (
            float(np.nanmean([rotation_jerk_subset(t, ~leaf_mask, feat_dim) for t in gt_pool]))
            if (gt_pool and leaf_mask is not None) else float('nan')
        )
        gt_pool_jerk_leaf = (
            float(np.nanmean([rotation_jerk_subset(t, leaf_mask, feat_dim) for t in gt_pool]))
            if (gt_pool and leaf_mask is not None) else float('nan')
        )
        # gt_intra_diversity: character-level mean intra-diversity over the GT pool.
        # Mirrors eval_truebones.py gt_intra_diversity_dist — baseline for intra_div_gt_diff.
        _gt_intra_vals = []
        for clip in (gt_pool or []):
            n_f = clip.shape[0]
            if n_f > 2 * tmin:
                _offsets = np.random.randint(n_f - tmin, size=(10, 2))
                for o in _offsets:
                    _gt_intra_vals.append(
                        avg_per_frame_dist(clip[o[0]:o[0]+tmin], clip[o[1]:o[1]+tmin], norm=feats)
                    )
        gt_intra_diversity = float(np.mean(_gt_intra_vals)) if _gt_intra_vals else float('nan')

        # Per-character adaptive coverage threshold: α × gt_intra_diversity.
        char_coverage_threshold = (
            coverage_adaptive_alpha * gt_intra_diversity
            if not math.isnan(gt_intra_diversity)
            else coverage_threshold
        )

        # Build GT pool with priority ordering:
        #   1. Source clips (ref/tgt for every pair) — always included
        #   2. Relevant clips from the benchmark sample lists (ref_samples / tgt_samples)
        #   3. Remaining GT clips up to MAX_GT_CLIPS
        # Paths are deduplicated; tensors are loaded once per unique path.
        def _load_clip(p):
            t = load_motion(p, source, feats, root_relative=root_relative)
            return t if t.shape[0] > 20 else None

        _source_paths = set()
        for _, meta in items:
            for key in ('ref_path', 'tgt_path'):
                p = resolve_gt_path(meta['samples'][key], gt_dir, source)
                if os.path.exists(p):
                    _source_paths.add(p)

        _relevant_paths = char_relevant_paths.get(char, set()) - _source_paths
        _remaining_paths = set(
            os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
            if f.endswith('.' + source)
            and os.path.splitext(f)[0].split('___')[0] == char
        ) - _source_paths - _relevant_paths

        char_pool = []
        seen = set()
        for p in sorted(_source_paths):
            if p not in seen:
                t = _load_clip(p)
                if t is not None:
                    char_pool.append(t)
                    seen.add(p)
        for p in sorted(_relevant_paths):
            if len(char_pool) >= MAX_GT_CLIPS:
                break
            if p not in seen and os.path.exists(p):
                t = _load_clip(p)
                if t is not None:
                    char_pool.append(t)
                    seen.add(p)
        for p in sorted(_remaining_paths):
            if len(char_pool) >= MAX_GT_CLIPS:
                break
            if p not in seen and os.path.exists(p):
                t = _load_clip(p)
                if t is not None:
                    char_pool.append(t)
                    seen.add(p)

        # Warn once per character if the pool has mixed joint dims (expected for x-skel).
        if char_pool:
            _pool_dims = set(c.shape[-1] for c in char_pool)
            if len(_pool_dims) > 1:
                print(f'  [{char}] pool has mixed joint dims {_pool_dims} '
                      f'— incompatible clips will be skipped per query.')

        char_results      = []   # metric dicts for this character
        char_tensors      = []   # blend_rot tensors for this character
        char_zone_tensors = []   # (pre, trans, post) slices per blend (None if zone too short)
        char_pair_keys    = []   # (ref, tgt, alpha) for this character
        char_same_skel    = []   # bool per result for this character

        for json_file, meta in items:
            print(f'  [{json_file}] ...')

            blend_path = resolve_blend_path(json_file, args.eval_gen_dir, source)
            ref_path   = resolve_gt_path(meta['samples']['ref_path'], gt_dir, source)
            tgt_path   = resolve_gt_path(meta['samples']['tgt_path'], gt_dir, source)

            missing = [p for p in (blend_path, ref_path, tgt_path) if not os.path.exists(p)]
            if missing:
                for p in missing:
                    print(f'  WARNING: file not found [{p}], skipping.')
                continue

            try:
                blend_rot = load_motion(blend_path, source, feats, root_relative=root_relative)
                ref_rot   = load_motion(ref_path,   source, feats, root_relative=root_relative)
                tgt_rot   = load_motion(tgt_path,   source, feats, root_relative=root_relative)
            except Exception as e:
                print(f'  WARNING: failed to load motion for [{json_file}]: {e}, skipping.')
                continue

            result = evaluate_blend_triple(
                blend_rot, meta, ref_rot, tgt_rot, source, feats,
                gt_pool=char_pool if char_pool else None,
                gt_pool_jerk=gt_pool_jerk,
                gt_pool_jerk_structural=gt_pool_jerk_structural,
                gt_pool_jerk_leaf=gt_pool_jerk_leaf,
                leaf_mask=leaf_mask,
                gt_intra_diversity=gt_intra_diversity,
                tmin=tmin,
            )
            result['sample'] = os.path.splitext(json_file)[0]
            del ref_rot, tgt_rot

            T  = blend_rot.shape[0]
            rr = meta['regions']
            _pre   = blend_rot[rr['ref_s']                  : min(rr['overlap_s'], T)]
            _trans = blend_rot[min(rr['overlap_s'], T)       : min(rr['overlap_e'], T)]
            _post  = blend_rot[min(rr['overlap_e'], T)       : min(rr['tgt_e'],    T)]
            char_zone_tensors.append((
                _pre   if _pre.shape[0]   >= tmin else None,
                _trans if _trans.shape[0] >= tmin else None,
                _post  if _post.shape[0]  >= tmin else None,
            ))

            char_results.append(result)
            char_tensors.append(blend_rot)
            char_pair_keys.append((
                meta['samples']['ref'],
                meta['samples']['tgt'],
                meta['blend']['alpha'],
            ))
            char_same_skel.append(result['same_skeleton'])

            all_results.append(result)
            all_same_skel.append(result['same_skeleton'])

        if not char_results:
            del gt_pool
            torch.cuda.empty_cache()
            continue

        # -- Inter-diversity for this character --------------------------------
        pair_to_local = defaultdict(list)
        for local_i, key in enumerate(char_pair_keys):
            pair_to_local[key].append(local_i)

        for key, local_indices in pair_to_local.items():
            if len(local_indices) < 2:
                continue
            blends     = [char_tensors[i] for i in local_indices]
            same_flags = [char_same_skel[i] for i in local_indices]
            for a in range(len(blends)):
                for b in range(a + 1, len(blends)):
                    if blends[a].shape[-1] != blends[b].shape[-1]:
                        continue  # skip cross-skeleton pairs with different joint counts
                    d = avg_per_frame_dist(blends[a], blends[b], norm=feats)
                    inter_div_all.append(d)
                    if same_flags[a] and same_flags[b]:
                        inter_div_in.append(d)

        # -- Coverage: fraction of GT windows covered by any blend for this char --
        if not math.isnan(gt_intra_diversity):
            gt_intra_div_all.append(gt_intra_diversity)

        if gt_pool:
            # Infer expected feature dim from the first blend tensor for this character.
            _expected_dim = char_tensors[0].shape[-1] if char_tensors else None
            gt_pool_trunc = [
                c for c in gt_pool[:MAX_GT_CLIPS]
                if _expected_dim is None or c.shape[-1] == _expected_dim
            ]

            # Per-pair coverage averaged over pairs — equal contribution per pair
            # regardless of how many reps were generated for each pair.
            def _pair_cov(zone_idx, same_skel_only=False):
                """Compute mean per-pair coverage for one zone (0=pre,1=trans,2=post)."""
                vals = []
                for key, local_indices in pair_to_local.items():
                    slices = [
                        char_zone_tensors[i][zone_idx]
                        for i in local_indices
                        if (not same_skel_only or char_same_skel[i])
                        and char_zone_tensors[i][zone_idx] is not None
                    ]
                    if slices:
                        by_dim = defaultdict(list)
                        for s in slices:
                            by_dim[s.shape[-1]].append(s)
                        for dim_slices in by_dim.values():
                            cat = torch.cat(dim_slices, dim=0)
                            _pool_compat = [c for c in gt_pool_trunc if c.shape[-1] == cat.shape[-1]]
                            if not _pool_compat:
                                continue
                            vals.append(coverage(
                                cat, _pool_compat,
                                threshold=char_coverage_threshold, tmin=tmin, use_pos=use_pos,
                            ))
                return float(np.mean(vals)) if vals else None

            pair_cov_all_vals = []
            pair_cov_in_vals  = []
            for key, local_indices in pair_to_local.items():
                # Group by feature dim to avoid cat errors across different skeletons
                by_dim = defaultdict(list)
                for i in local_indices:
                    by_dim[char_tensors[i].shape[-1]].append(i)
                for dim_indices in by_dim.values():
                    pair_blends = torch.cat([char_tensors[i] for i in dim_indices], dim=0)
                    _pool_compat = [c for c in gt_pool_trunc if c.shape[-1] == pair_blends.shape[-1]]
                    if not _pool_compat:
                        continue
                    pair_cov_all_vals.append(coverage(
                        pair_blends, _pool_compat,
                        threshold=char_coverage_threshold, tmin=tmin, use_pos=use_pos,
                    ))
                    in_blends = [char_tensors[i] for i in dim_indices if char_same_skel[i]]
                    if in_blends:
                        pair_cov_in_vals.append(coverage(
                            torch.cat(in_blends, dim=0), _pool_compat,
                            threshold=char_coverage_threshold, tmin=tmin, use_pos=use_pos,
                        ))

            if pair_cov_all_vals:
                coverage_all.append(float(np.mean(pair_cov_all_vals)))
            if pair_cov_in_vals:
                coverage_in.append(float(np.mean(pair_cov_in_vals)))

            for zone_idx, (acc_all, acc_in) in enumerate([
                (coverage_pre_all,   coverage_pre_in),
                (coverage_trans_all, coverage_trans_in),
                (coverage_post_all,  coverage_post_in),
            ]):
                v = _pair_cov(zone_idx, same_skel_only=False)
                if v is not None:
                    acc_all.append(v)
                v = _pair_cov(zone_idx, same_skel_only=True)
                if v is not None:
                    acc_in.append(v)

        del char_tensors, gt_pool
        torch.cuda.empty_cache()

    # ------- Aggregate --------------------------------------------------------
    if not all_results:
        print('No results to aggregate.')
        return {}

    output = {}

    _nan_dict = {'mean': float('nan'), 'std': float('nan'),
                 'median': float('nan'), 'iqr': float('nan'), 'n': 0}

    all_dict = aggregate_results(all_results)
    all_dict['inter_diversity']    = _agg(inter_div_all)
    all_dict['gt_intra_diversity'] = _agg(gt_intra_div_all)
    all_dict['coverage']           = _cov_dict(coverage_all)
    all_dict['coverage_pre']       = _cov_dict(coverage_pre_all)
    all_dict['coverage_trans']     = _cov_dict(coverage_trans_all)
    all_dict['coverage_post']      = _cov_dict(coverage_post_all)
    print_results('ALL', all_dict)
    output['ALL'] = all_dict

    in_skel_results = [r for r, s in zip(all_results, all_same_skel) if s]
    if in_skel_results:
        in_dict = aggregate_results(in_skel_results)
        in_dict['inter_diversity']    = _agg(inter_div_in)
        in_dict['gt_intra_diversity'] = _agg(gt_intra_div_all)
        in_dict['coverage']           = _cov_dict(coverage_in)
        in_dict['coverage_pre']       = _cov_dict(coverage_pre_in)
        in_dict['coverage_trans']     = _cov_dict(coverage_trans_in)
        in_dict['coverage_post']      = _cov_dict(coverage_post_in)
        print_results('IN_SKEL', in_dict)
        output['IN_SKEL'] = in_dict

    x_skel_results = [r for r, s in zip(all_results, all_same_skel) if not s]
    if x_skel_results:
        x_dict = aggregate_results(x_skel_results)
        x_dict['inter_diversity']    = _nan_dict
        x_dict['gt_intra_diversity'] = _agg(gt_intra_div_all)
        x_dict['coverage']           = _nan_dict
        x_dict['coverage_pre']       = _nan_dict
        x_dict['coverage_trans']     = _nan_dict
        x_dict['coverage_post']      = _nan_dict
        print_results('X_SKEL', x_dict)
        output['X_SKEL'] = x_dict

    output['raw'] = {
        'samples': [r['sample'] for r in all_results],
        'metrics': [{k: v for k, v in r.items() if k != 'sample'} for r in all_results],
    }
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    print(f'SOURCE=[{args.source}]  FEATS=[{args.feats}]  ROOT_RELATIVE=[{args.root_relative}]')

    assert os.path.isdir(args.eval_gen_dir), f'Invalid gen dir [{args.eval_gen_dir}]'
    _resolved_gt = gt_dir_for_source(args.eval_gt_dir, args.source)
    assert os.path.isdir(_resolved_gt), (
        f'Invalid gt dir [{_resolved_gt}] (resolved from [{args.eval_gt_dir}] + source [{args.source}])'
    )

    bench_name = os.path.basename(os.path.normpath(args.eval_gen_dir))
    out_stem   = os.path.join(
        os.path.dirname(os.path.normpath(args.eval_gen_dir)),
        bench_name + f'__source_{args.source}__feats_{args.feats}' + ('__rootrel' if args.root_relative else '') + args.unique_str,
    )
    json_file = out_stem + '.json'
    print(f'Will save to [{json_file}]')

    eval_dict = eval_blend_benchmark(args)

    def _json_safe(obj):
        """Recursively replace non-JSON-serialisable floats (nan/inf) with None."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with open(json_file, 'w') as fw:
        json_module.dump(_json_safe(eval_dict), fw, indent=2)
