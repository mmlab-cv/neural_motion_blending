"""
fid_truebones_blend.py — Per-pair FID in MoDiffAE semantic latent space.

GT distribution  : latent codes of the GT ref/tgt clip(s).
Gen distribution : latent codes from all repetitions of a generated pair.
Aggregated per-character (mean ± stderr) and globally (mean ± stderr + median).

GT latents are cached under:
    <parent of eval_gen_dir>/gt_latent_cache/<model_stem>/<clip_stem>.npz

Generated motions can be provided as:
  npy  — processed .npy files (direct output of sample/mix.py)
  bvh  — BVH files (sample/nla_blend.py, motion2motion, etc.)
          Must be already in HML space (as produced by nla_blend/process_object).
          Joint order is permuted back to cond_dict canonical order before encoding.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import scipy.linalg

from utils import dist_util
from utils.parser_util import add_base_options, add_model_options, extract_args, get_args_per_group_name
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
import BVH
from data_loaders.truebones.truebones_utils.motion_process import (
    get_bvh_cont6d_params,
    get_foot_contact,
    get_rifke,
    get_motion_features,
    FOOT_CONTACT_VEL_THRESH,
)
from model.motion_diffusion_ae import gather_vars
from model.modules.conditioners import T5Conditioner
from sample.mix import create_sample_in_batch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    from argparse import ArgumentParser
    p = ArgumentParser(description='Compute per-pair FID in MoDiffAE latent space.')
    add_base_options(p)
    add_model_options(p)

    p.add_argument('--eval_gen_dir', required=True,
                   help='Directory with generated motion files (+ JSON metadata).')
    p.add_argument('--eval_gt_dir', required=True,
                   help='Base dataset directory, or the motions/ subdir directly.')
    p.add_argument('--model_path',
                   default='save/modiffae_truebones_all_globpool/model000449998.pt',
                   help='Path to MoDiffAE checkpoint. Model arch args are loaded from '
                        'the args.json in the same directory.')
    p.add_argument('--source', choices=['npy', 'bvh'], default='npy',
                   help='Format of generated files. GT clips are always loaded as npy.')
    p.add_argument('--bvh_ref_dir', default='',
                   help='(Unused — BVH feature extraction no longer needs T-pose re-derivation.)')
    p.add_argument('--cond_path', default='',
                   help='Path to cond.npy (defaults to dataset default).')
    p.add_argument('--gt_cache_dir', default='',
                   help='Dir to cache GT latents. '
                        'Defaults to <parent of eval_gen_dir>/gt_latent_cache/<model_stem>/.')
    p.add_argument('--output_json', default='',
                   help='Write full results to this JSON file.')
    p.add_argument('--gen_frames', choices=['all', 'transition', 'zones'], default='transition',
                   help='"all": use entire generated clip (gt_mode controls GT). '
                        '"transition": use only [overlap_s, overlap_e] frames (gt_mode controls GT). '
                        '"zones": report three separate FIDs — pre [ref_s:overlap_s] vs ref GT, '
                        'transition [overlap_s:overlap_e] vs gt_mode GT, '
                        'post [overlap_e:tgt_e] vs tgt GT. Overrides gt_mode for pre/post zones.')
    p.add_argument('--gt_mode', choices=['tgt', 'ref', 'both'], default='tgt',
                   help='GT reference distribution: target clip, ref clip, or both concatenated. '
                        'Used for the transition zone (or whole clip) in all gen_frames modes.')
    p.add_argument('--force_recompute', action='store_true',
                   help='Ignore existing GT latent cache and recompute from scratch.')
    p.add_argument('--zero_root_translation', action='store_true',
                   help='Zero out root translation features (root joint velocity feat[:,0,9:12] and '
                        'root Y position feat[:,0,1]) before encoding, for both GT and generated '
                        'samples. Removes locomotion direction/speed from the latent comparison.')
    p.add_argument('--gt_window_size', type=float, default=2.0,
                   help='Controls the GT window for the transition zone. '
                        'Positive value: multiplier over the detected overlap length — '
                        'e.g. 1.0 takes exactly overlap_length frames per side, 2.0 takes twice as many. '
                        'Both sides are capped to min(N, available frames) and kept equal. '
                        'Negative (default -1): use the full clip on each side (no windowing).')

    args = p.parse_args()
    # Load model arch args from args.json alongside the checkpoint (same as mix.py).
    # This includes t5_name, latent_dim, num_virtual_joints, etc. — all must match training.
    args = extract_args(args, get_args_per_group_name(p, args, 'model'), args.model_path)
    return args


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(z1, z2, eps=1e-6):
    """FID between two (N, C) float arrays. Returns nan if either has < 2 samples."""
    if z1.shape[0] < 2 or z2.shape[0] < 2:
        return float('nan')
    mu1, mu2 = z1.mean(0), z2.mean(0)
    sig1, sig2 = np.cov(z1, rowvar=False), np.cov(z2, rowvar=False)
    if sig1.ndim == 0: sig1 = sig1.reshape(1, 1)
    if sig2.ndim == 0: sig2 = sig2.reshape(1, 1)
    covmean, _ = scipy.linalg.sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = scipy.linalg.sqrtm((sig1 + eps * np.eye(sig1.shape[0])) @
                                     (sig2 + eps * np.eye(sig2.shape[0])))
    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-3:
            print('    WARNING: large imaginary component in sqrtm — discarding.')
        covmean = covmean.real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(sig1 + sig2 - 2.0 * covmean))


def agg(values):
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    return (float(np.mean(arr)),
            float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
            float(np.median(arr)), n)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(model, batch, model_kwargs, device):
    """Returns list of (L_i, C) tensors, one per sample, padding stripped."""
    x = batch.to(device)
    y = {k: (v.to(device) if torch.is_tensor(v) else v)
         for k, v in model_kwargs['y'].items()}

    x_ctrl = x.unsqueeze(1)  # (B, 1, J, F, T)
    topo_rel, edge_rel, j_mask, t_mask, tpos_ff = gather_vars(x_ctrl, y)
    z_sem, _ = model.semantic_encoder(
        x_ctrl, y, topo_rel, edge_rel, j_mask, t_mask, tpos_ff,
        alpha=None, get_layer_activation=[],
    )

    # Normalise shape to (f, B, C)
    if z_sem.dim() == 5 and z_sem.shape[2] == 1 and z_sem.shape[3] == 1:
        z_sem = z_sem[:, :, 0, 0, :]          # pool path: (f,B,1,1,d) → (f,B,d)
    elif z_sem.dim() == 5:
        z_sem = z_sem[:, :, :, 0, :].mean(0)  # virtual joints: (V,f,B,1,d) → (f,B,d)
    else:
        raise ValueError(f'Unexpected z_sem shape: {z_sem.shape}')

    z_sem = z_sem[1:].permute(1, 0, 2)  # drop T-pose frame → (B, T, C)
    lengths = y['lengths']
    return [z_sem[i, :int(lengths[i]), :] for i in range(x.shape[0])]


@torch.no_grad()
def encode_path(path, source, object_type, cond_dict, t5_conditioner, opt, model, device,
                temporal_window=31, zero_root_trans=False):
    """Load one motion file and return its latent codes as (N_frames, C) numpy array."""
    b, kw = load_clip_as_batch(path, source, object_type, cond_dict, t5_conditioner, opt,
                               temporal_window, zero_root_trans=zero_root_trans)
    if b is None:
        return None
    out = encode_batch(model, b, kw, device)
    return out[0].cpu().float().numpy() if out else None


# ---------------------------------------------------------------------------
# GT latent cache
# ---------------------------------------------------------------------------

def load_gt_latents(clip_path, object_type, gt_cache_dir,
                    cond_dict, t5_conditioner, opt, model, device,
                    zero_root_trans=False, force_recompute=False):
    """Return (N_frames, C) GT latents; encode + cache on first call."""
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    suffix = '_noroottrans' if zero_root_trans else ''
    cache = os.path.join(gt_cache_dir, stem + suffix + '.npz')
    if os.path.exists(cache) and not force_recompute:
        return np.load(cache)['z']

    z = encode_path(clip_path, 'npy', object_type, cond_dict, t5_conditioner, opt,
                    model, device, zero_root_trans=zero_root_trans)
    if z is None:
        print(f'    WARNING: could not encode GT clip [{clip_path}]')
        return None

    os.makedirs(gt_cache_dir, exist_ok=True)
    np.savez_compressed(cache, z=z)
    print(f'    GT cached → [{cache}]  ({z.shape[0]} frames)')
    return z


# ---------------------------------------------------------------------------
# BVH → features
# ---------------------------------------------------------------------------


def _reorder_anim(anim, bvh_names, ref_names, ocd=None):
    """Permute joints of an Animation to match ref_names ordering.

    Handles three sources of mismatch:
    - Blender reorders bones on BVH export (same names, different order)
    - motion2motion appends a uniqueness suffix '__XXX' to joint names
    - Named End Sites in the GT BVH are promoted to joints by BVH.py but
      motion2motion omits them (they carry no rotation DOF). When ocd is
      provided, these are inserted with identity rotations and offsets taken
      from cond_dict so that the joint count matches.

    Suffix stripping is attempted only when direct name lookup fails.
    Only the rightmost '__'-separated suffix is stripped, since '__' may
    appear legitimately in the base name.
    """
    from Animation import Animation
    from Quaternions import Quaternions

    if bvh_names == ref_names:
        return anim  # already in correct order

    # ── Step 1: build name → BVH-index map, with optional suffix stripping ──
    name_to_bvh = {n: i for i, n in enumerate(bvh_names)}
    missing = [n for n in ref_names if n not in name_to_bvh]
    if missing:
        def _strip(name):
            parts = name.rsplit('__', 1)
            return parts[0] if len(parts) == 2 and parts[0] else name
        stripped = [_strip(n) for n in bvh_names]
        name_to_bvh = {s: i for i, s in enumerate(stripped)}
        missing = [n for n in ref_names if n not in name_to_bvh]

    # ── Step 2: insert missing End-Site joints with identity rotations ──
    if missing:
        if ocd is None:
            raise ValueError(f'BVH is missing joints present in cond_dict: {missing}')

        ref_parents  = list(ocd['parents'])      # parent index per ref joint
        ref_offsets  = np.array(ocd['offsets'])  # (J, 3) rest offsets
        T, J_bvh = anim.rotations.shape   # Quaternions shape is (T, J) — 4th dim hidden
        n_missing    = len(missing)
        missing_set  = set(missing)

        # Verify all missing joints are leaves in the ref skeleton (no ref joint
        # has one of the missing joints as its parent — End Sites have no children).
        ref_name_to_idx = {n: i for i, n in enumerate(ref_names)}
        for m in missing:
            m_idx = ref_name_to_idx[m]
            children = [i for i, p in enumerate(ref_parents) if p == m_idx]
            if children:
                child_names = [ref_names[c] for c in children]
                raise ValueError(
                    f'BVH is missing non-leaf joint "{m}" '
                    f'(has children {child_names} in cond_dict skeleton)'
                )

        print(f'    INFO: inserting {n_missing} missing End-Site joint(s) with identity '
              f'rotations: {missing}')

        # Extend BVH arrays with identity-rotation columns for each missing joint.
        # Their parent in ref_names is used to find the correct parent index in
        # the (post-extension) bvh_names list.
        bvh_names    = list(bvh_names)
        new_rots_ext = np.zeros((T, n_missing, 4))
        new_rots_ext[..., 3] = 1.0           # identity quaternion w=1
        new_pos_ext  = np.zeros((T, n_missing, 3))
        new_off_ext  = np.zeros((n_missing, 3))
        new_ori_ext  = np.zeros((n_missing, 4))
        new_ori_ext[..., 3] = 1.0
        new_par_ext  = []

        for m in missing:
            m_idx     = ref_name_to_idx[m]
            ref_par   = ref_parents[m_idx]           # parent idx in ref skeleton
            par_name  = ref_names[ref_par] if ref_par != -1 else None
            # Find this parent in the (possibly suffix-stripped) bvh name map
            par_bvh   = name_to_bvh.get(par_name, -1) if par_name is not None else -1
            new_par_ext.append(par_bvh)
            new_off_ext[missing.index(m)] = ref_offsets[m_idx]
            bvh_names.append(m)

        anim_rots = Quaternions(np.concatenate([anim.rotations.qs, new_rots_ext], axis=1))
        anim_pos  = np.concatenate([anim.positions, new_pos_ext],  axis=1)
        anim_off  = np.concatenate([anim.offsets,   new_off_ext],  axis=0)
        anim_ori  = Quaternions(np.concatenate([anim.orients.qs,   new_ori_ext],  axis=0))
        anim_par  = np.concatenate([anim.parents,   new_par_ext])

        # Rebuild the animation with extended arrays so the reorder below works.
        anim = Animation(anim_rots, anim_pos, anim_ori, anim_off, anim_par)
        # Add the newly inserted joints to the existing (possibly suffix-stripped) map.
        # The original joints already have their (stripped) names in name_to_bvh;
        # the newly appended ones are stored at indices J_bvh, J_bvh+1, ... and
        # their names are the clean ref_names (no suffix), so add them directly.
        for k, m in enumerate(missing):
            name_to_bvh[m] = J_bvh + k
        still_missing = [n for n in ref_names if n not in name_to_bvh]
        if still_missing:
            raise ValueError(f'BVH is still missing joints after End-Site insertion: {still_missing}')

    # ── Step 3: permute to ref_names order ──
    perm = np.array([name_to_bvh[n] for n in ref_names])

    bvh_to_new = {old: new for new, old in enumerate(perm)}
    new_parents = np.array([
        bvh_to_new[anim.parents[perm[i]]] if anim.parents[perm[i]] != -1 else -1
        for i in range(len(ref_names))
    ])

    new_rots      = anim.rotations[:, perm]
    new_positions = anim.positions[:, perm]
    new_offsets   = anim.offsets[perm]
    new_orients   = anim.orients[perm]

    return Animation(new_rots, new_positions, new_orients, new_offsets, new_parents)


from utils.bvh_io import bvh_load_safe as _bvh_load_safe


def bvh_to_features(bvh_path, object_type, cond_dict):
    """Convert a BVH file to motion features.

    The BVH files produced by nla_blend / motion2motion / process_object are *already*
    in HML space — rotations are T-pose-relative, positions are scaled and grounded.
    Re-running process_anim on them would double-transform the data.
    Instead, load the animation and extract features directly, mirroring what
    get_motion() does after get_hml_aligned_anim() returns.

    Blender can reorder joints on BVH export; we reorder back to the canonical
    ordering from cond_dict before extracting features.
    """
    ocd = cond_dict.get(object_type, {})
    if not ocd:
        print(f'  WARNING: "{object_type}" not in cond_dict')
        return None
    foot_indices = ocd['foot_indices']
    face_joints  = ocd['face_joints']
    ref_names    = list(ocd['joints_names'])
    max_joints   = len(ocd['offsets'])

    try:
        anim, bvh_names, _ = _bvh_load_safe(bvh_path)
        anim = _reorder_anim(anim, list(bvh_names), ref_names, ocd=ocd)
        cont_6d_params, _, _, r_rot, global_positions = \
            get_bvh_cont6d_params(anim, object_type, face_joints=face_joints)
        foot_contact = get_foot_contact(global_positions, foot_indices, FOOT_CONTACT_VEL_THRESH)
        positions = get_rifke(global_positions, r_rot)
        local_vel = (np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1)
                     * (global_positions[1:] - global_positions[:-1]))
        features, _ = get_motion_features(positions, cont_6d_params, foot_contact,
                                          local_vel, max_joints)
        return features
    except Exception as e:
        print(f'  WARNING: failed to extract features from [{bvh_path}]: {e}')
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npy_features(npy_path):
    try:
        data = np.load(npy_path, allow_pickle=True)
        if data.dtype == object:
            data = data.item()['motion_raw'].transpose(0, 3, 1, 2)
        return data
    except Exception as e:
        print(f'  WARNING: failed to load [{npy_path}]: {e}')
        return None


def features_to_batch(features, object_type, cond_dict, t5_conditioner, opt, temporal_window=31):
    if features is None or features.shape[0] < 2:
        return None, None
    if object_type not in cond_dict:
        print(f'  WARNING: "{object_type}" not in cond_dict — skipping.')
        return None, None

    ocd = cond_dict[object_type]
    if features.shape[1] != len(ocd['parents']):
        print(f'  WARNING: joint count mismatch for "{object_type}": '
              f'features={features.shape[1]}, cond={len(ocd["parents"])} — skipping clip. '
              f'The BVH skeleton does not match the dataset skeleton for this character.')
        return None, None

    sample = create_sample_in_batch(
        object_type, ocd, features.shape[0], opt.max_joints, opt.feature_len,
        temporal_window=temporal_window, t5_conditioner=t5_conditioner,
        motion=features, frame_start=0, loop_times=1,
    )
    return truebones_batch_collate([sample])


def zero_root_translation(features):
    """Zero root joint translation from features (F, J, 13).

    Root XZ position is already 0 by construction (get_rifke). This zeros:
      - feat[:, 0, 1]    : root Y position (height variation)
      - feat[:, 0, 9:12] : root local velocity (locomotion direction/speed)
    """
    features = features.copy()
    features[:, 0, 1]    = 0.0
    features[:, 0, 9:12] = 0.0
    return features


def load_clip_as_batch(path, source, object_type, cond_dict, t5_conditioner, opt,
                       temporal_window=31, zero_root_trans=False):
    features = (load_npy_features(path) if source == 'npy'
                else bvh_to_features(path, object_type, cond_dict))
    if zero_root_trans and features is not None:
        features = zero_root_translation(features)
    return features_to_batch(features, object_type, cond_dict, t5_conditioner, opt,
                             temporal_window)


def resolve_gt_npy(field, meta, gt_npy_dir):
    """Resolve a GT clip to its .npy path regardless of how the JSON stores it."""
    stem = os.path.splitext(os.path.basename(meta['samples'][field]))[0]
    p = os.path.join(gt_npy_dir, stem + '.npy')
    return p if os.path.exists(p) else meta['samples'][field + '_path']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # dist_util.setup_dist expects an int device index
    dev_idx = args.device if isinstance(args.device, int) else (0 if args.device != 'cpu' else -1)
    dist_util.setup_dist(dev_idx)
    device = dist_util.dev()

    opt = get_opt(args.device)

    # Resolve paths
    cond_path = args.cond_path or opt.cond_file
    cond_dict = np.load(cond_path, allow_pickle=True).item()
    print(f'Loaded cond.npy from [{cond_path}]')

    gt_npy_dir = os.path.normpath(args.eval_gt_dir)
    if os.path.basename(gt_npy_dir) != 'motions':
        candidate = os.path.join(gt_npy_dir, 'motions')
        if os.path.isdir(candidate):
            gt_npy_dir = candidate

    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    gt_cache_dir = (args.gt_cache_dir or
                    os.path.join(os.path.dirname(os.path.abspath(args.eval_gen_dir)),
                                 'gt_latent_cache', model_stem))
    print(f'GT npy dir   : [{gt_npy_dir}]')
    print(f'GT cache dir : [{gt_cache_dir}]')

    # Load model
    print('Loading T5 conditioner ...')
    t5_conditioner = T5Conditioner(
        name=args.t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device='cuda',
    )
    print(f'Loading MoDiffAE from [{args.model_path}] ...')
    model, _ = create_model_and_diffusion_general_skeleton(args)
    load_model(model, torch.load(args.model_path, map_location='cpu', weights_only=False))
    model.to(device)
    model.eval()

    # Parse JSONs and group by (ref_path, tgt_path) from JSON content
    ext = '.npy' if args.source == 'npy' else '.bvh'
    json_files = sorted(f for f in os.listdir(args.eval_gen_dir) if f.endswith('.json'))
    if not json_files:
        print(f'No JSON files found in [{args.eval_gen_dir}].')
        return

    pair_gen_paths = defaultdict(list)
    pair_meta = {}
    for jf in json_files:
        with open(os.path.join(args.eval_gen_dir, jf)) as fp:
            meta = json.load(fp)
        pair_key = (meta['samples']['ref_path'], meta['samples']['tgt_path'])
        gen_path = os.path.join(args.eval_gen_dir, os.path.splitext(jf)[0] + ext)
        if os.path.exists(gen_path):
            pair_gen_paths[pair_key].append(gen_path)
        if pair_key not in pair_meta:
            pair_meta[pair_key] = meta

    print(f'\nFound {len(pair_gen_paths)} unique (ref, tgt) pairs '
          f'across {len(json_files)} JSON files.\n')

    zones_mode = (args.gen_frames == 'zones')

    # Per-pair FID — three parallel accumulators when zones_mode is active
    # keys: 'main' (default), 'pre', 'post'
    char_fids  = defaultdict(lambda: defaultdict(list))
    all_fids   = defaultdict(list)
    pair_results = []

    for pair_key in sorted(pair_gen_paths.keys()):
        meta       = pair_meta[pair_key]
        ref_object = meta['samples']['ref_object']
        tgt_object = meta['samples']['tgt_object']
        gen_paths  = pair_gen_paths[pair_key]

        local_tgt = resolve_gt_npy('tgt', meta, gt_npy_dir)
        local_ref = resolve_gt_npy('ref', meta, gt_npy_dir)

        r         = meta.get('regions', {})
        ref_s     = r.get('ref_s',     0)
        overlap_s = r.get('overlap_s', 0)
        overlap_e = r.get('overlap_e', 0)
        tgt_e     = r.get('tgt_e',     overlap_e)
        has_transition = overlap_e > overlap_s
        has_pre        = overlap_s > ref_s
        has_post       = tgt_e > overlap_e

        print(f'Pair: {os.path.basename(meta["samples"]["ref"])} → '
              f'{os.path.basename(meta["samples"]["tgt"])}  '
              f'({len(gen_paths)} rep(s))'
              + (f'  [pre {ref_s}:{overlap_s} | trans {overlap_s}:{overlap_e} | post {overlap_e}:{tgt_e}]'
                 if has_transition else ''))

        # Encode GT clips (cached)
        def _gt(clip_path, obj):
            return load_gt_latents(clip_path, obj, gt_cache_dir,
                                   cond_dict, t5_conditioner, opt, model, device,
                                   zero_root_trans=args.zero_root_translation,
                                   force_recompute=args.force_recompute)

        z_gt_ref = _gt(local_ref, ref_object)
        z_gt_tgt = _gt(local_tgt, tgt_object)

        def _gt_for_mode(mode, n_frames=None):
            """Build GT distribution for the transition zone.

            If n_frames is None: use the full clip(s).
            If n_frames is given: use a balanced window of at most n_frames per side —
              - ref: last n frames before overlap_s  →  z_gt_ref[overlap_s-n : overlap_s]
              - tgt: first n frames from clip start  →  z_gt_tgt[0 : n]
            The actual n is min(n_frames, available ref frames, available tgt frames)
            so both sides always contribute equally.
            """
            if n_frames is None:
                parts = []
                if mode != 'ref' and z_gt_tgt is not None: parts.append(z_gt_tgt)
                if mode != 'tgt' and z_gt_ref is not None: parts.append(z_gt_ref)
                return np.concatenate(parts, axis=0) if parts else None

            # Compute how many frames each side can actually provide, then balance
            avail_ref = (min(overlap_s, len(z_gt_ref)) - ref_s) if (mode != 'tgt' and z_gt_ref is not None) else n_frames
            avail_tgt = len(z_gt_tgt)                           if (mode != 'ref' and z_gt_tgt is not None) else n_frames
            n = min(n_frames, avail_ref, avail_tgt)

            parts = []
            if mode != 'ref' and z_gt_tgt is not None:
                parts.append(z_gt_tgt[:n])
            if mode != 'tgt' and z_gt_ref is not None:
                parts.append(z_gt_ref[overlap_s - n : overlap_s])
            return np.concatenate(parts, axis=0) if parts else None

        # Encode generated clips (all reps)
        z_encoded = []
        for gp in sorted(gen_paths):
            z = encode_path(gp, args.source, ref_object,
                            cond_dict, t5_conditioner, opt, model, device,
                            zero_root_trans=args.zero_root_translation)
            if z is not None:
                z_encoded.append(z)

        if not z_encoded:
            print('  SKIP: no generated clips could be encoded.')
            continue

        def _cat_zone(z_list, s, e):
            slices = [z[s:e] for z in z_list if len(z[s:e]) > 0]
            return np.concatenate(slices, axis=0) if slices else np.empty((0, z_list[0].shape[-1]))

        pair_entry = {
            'ref': meta['samples']['ref'], 'tgt': meta['samples']['tgt'],
            'ref_object': ref_object, 'tgt_object': tgt_object,
            'n_reps': len(z_encoded),
        }

        if zones_mode:
            # --- pre zone: gen[ref_s:overlap_s] vs ref GT[ref_s:overlap_s] ---
            if has_pre and z_gt_ref is not None:
                z_pre    = _cat_zone(z_encoded, ref_s, overlap_s)
                z_gt_pre = z_gt_ref[ref_s:overlap_s]
                fid_pre  = compute_fid(z_gt_pre, z_pre) if (z_pre.shape[0] >= 2 and z_gt_pre.shape[0] >= 2) else float('nan')
                char_fids[ref_object]['pre'].append(fid_pre)
                all_fids['pre'].append(fid_pre)
                pair_entry['fid_pre'] = fid_pre
                print(f'  pre  FID = {fid_pre:.4f}  '
                      f'(gt_ref={z_gt_pre.shape[0]}, gen={z_pre.shape[0]} frames)')

            # --- transition zone: gen[overlap_s:overlap_e] vs gt_mode GT ---
            if has_transition:
                trans_len  = overlap_e - overlap_s
                gt_n       = int(args.gt_window_size * trans_len) if args.gt_window_size >= 0 else None
                z_gt_trans = _gt_for_mode(args.gt_mode, n_frames=gt_n)
                if z_gt_trans is not None and z_gt_trans.shape[0] >= 2:
                    z_trans = _cat_zone(z_encoded, overlap_s, overlap_e)
                    fid_trans = compute_fid(z_gt_trans, z_trans) if z_trans.shape[0] >= 2 else float('nan')
                    char_fids[ref_object]['transition'].append(fid_trans)
                    all_fids['transition'].append(fid_trans)
                    pair_entry['fid_transition'] = fid_trans
                    print(f'  trans FID = {fid_trans:.4f}  '
                          f'(gt={z_gt_trans.shape[0]}, gen={z_trans.shape[0]} frames)')

            # --- post zone: gen[overlap_e:tgt_e] vs tgt GT[overlap_e-tgt_s:tgt_e-tgt_s] ---
            if has_post and z_gt_tgt is not None:
                z_post    = _cat_zone(z_encoded, overlap_e, tgt_e)
                tgt_s_val = r.get('tgt_s', 0)
                z_gt_post = z_gt_tgt[overlap_e - tgt_s_val : tgt_e - tgt_s_val]
                fid_post  = compute_fid(z_gt_post, z_post) if (z_post.shape[0] >= 2 and z_gt_post.shape[0] >= 2) else float('nan')
                char_fids[ref_object]['post'].append(fid_post)
                all_fids['post'].append(fid_post)
                pair_entry['fid_post'] = fid_post
                print(f'  post FID = {fid_post:.4f}  '
                      f'(gt_tgt={z_gt_post.shape[0]}, gen={z_post.shape[0]} frames)')

        else:
            is_trans_mode = args.gen_frames == 'transition' and has_transition
            if is_trans_mode and args.gt_window_size >= 0:
                trans_len = overlap_e - overlap_s
                gt_n = int(args.gt_window_size * trans_len)
            else:
                gt_n = None
            z_gt   = _gt_for_mode(args.gt_mode, n_frames=gt_n)
            if z_gt is None or z_gt.shape[0] < 2:
                print('  SKIP: GT has too few frames.')
                continue

            z_gen = (_cat_zone(z_encoded, overlap_s, overlap_e)
                     if is_trans_mode
                     else np.concatenate(z_encoded, axis=0))

            if z_gen.shape[0] < 2:
                print(f'  SKIP: too few generated frames ({z_gen.shape[0]}).')
                continue

            fid = compute_fid(z_gt, z_gen)
            print(f'  FID = {fid:.4f}  '
                  f'(gt={z_gt.shape[0]} frames, gen={z_gen.shape[0]} frames, '
                  f'{len(z_encoded)} rep(s))')

            char_fids[ref_object]['main'].append(fid)
            all_fids['main'].append(fid)
            pair_entry.update({
                'fid': fid,
                'gt_frames': int(z_gt.shape[0]), 'gen_frames': int(z_gen.shape[0]),
                'gt_mode': args.gt_mode, 'gen_frames_mode': args.gen_frames,
            })

        pair_results.append(pair_entry)

    # Aggregation
    zone_labels = (['pre', 'transition', 'post'] if zones_mode else ['main'])
    display_labels = ({'pre': 'PRE', 'transition': 'TRANSITION', 'post': 'POST', 'main': ''}
                      if zones_mode else {'main': ''})

    for zone in zone_labels:
        label = display_labels[zone]
        header = f'PER-CHARACTER FID {label} (mean ± stderr over pairs):'
        print(f'\n{"=" * 60}')
        print(header)
        char_summary_zone = {}
        for char in sorted(char_fids):
            vals = char_fids[char][zone]
            if not vals:
                continue
            mean, stderr, median, n = agg(vals)
            char_summary_zone[char] = {'mean': mean, 'stderr': stderr, 'median': median, 'n': n}
            print(f'  {char:30s}  {mean:.4f} ± {stderr:.4f}  (median={median:.4f}, n={n})')

        g_mean, g_stderr, g_median, g_n = agg(all_fids[zone])
        print(f'\nGLOBAL {label} FID  (over {g_n} pairs):')
        print(f'  Mean   : {g_mean:.4f} ± {g_stderr:.4f}')
        print(f'  Median : {g_median:.4f}')

    if args.output_json:
        out = {
            'zones': zone_labels,
            'global': {zone: dict(zip(['mean','stderr','median','n_pairs'], agg(all_fids[zone])))
                       for zone in zone_labels},
            'per_pair': pair_results,
            'model_path': args.model_path,
            'source': args.source,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as fp:
            json.dump(out, fp, indent=4)
        print(f'\nResults written to [{args.output_json}]')


if __name__ == '__main__':
    main()
