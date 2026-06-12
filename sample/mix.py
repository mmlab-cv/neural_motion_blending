# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import json
import random
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from copy import deepcopy
from utils.parser_util import mix_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from utils.misc import char_from_npy_stem
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from model.modules.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from model.motion_diffusion_ae import ControlConfig

from model.modules.util.geom import slerp
from model.motion_diffusion_ae import MoDiffAE

from einops import rearrange

def _purge_ghost_outputs(out_path):
    """
    Delete any .npy, .bvh, or .mp4 file in out_path whose stem has no
    matching .json file. These are partial outputs from a run that was
    interrupted before the JSON could be written.
    """
    if not os.path.isdir(out_path):
        return
    json_stems = {f[:-5] for f in os.listdir(out_path) if f.endswith('.json')}
    for fname in os.listdir(out_path):
        ext = os.path.splitext(fname)[1]
        if ext not in ('.npy', '.bvh', '.mp4'):
            continue
        stem = os.path.splitext(fname)[0]
        if stem not in json_stems:
            ghost_path = os.path.join(out_path, fname)
            print(f"  [resume] removing ghost file: {fname}")
            os.remove(ghost_path)


def _scan_completed_pairs(out_path):
    """
    Read every JSON in out_path and build a dict:
        (ref_basename, tgt_basename) -> count of outputs that also have
        companion .npy and .bvh files present on disk.

    Keys use the bare filename stored in meta['samples']['ref'/'tgt']
    (e.g. "Alligator___Walk3_22.npy"), which avoids relative/absolute
    path mismatches between the JSON and the input txt lists.
    """
    from collections import defaultdict
    counts = defaultdict(int)
    if not os.path.isdir(out_path):
        return counts
    for fname in os.listdir(out_path):
        if not fname.endswith('.json'):
            continue
        json_path = os.path.join(out_path, fname)
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            ref_base = meta['samples']['ref']   # bare filename, e.g. "Alligator___Walk3_22.npy"
            tgt_base = meta['samples']['tgt']
        except Exception:
            continue
        stem = fname[:-5]  # strip .json
        npy_ok = os.path.isfile(os.path.join(out_path, stem + '.npy'))
        bvh_ok = os.path.isfile(os.path.join(out_path, stem + '.bvh'))
        if npy_ok and bvh_ok:
            counts[(ref_base, tgt_base)] += 1
    return counts


def main(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = mix_args()
    fixseed(args.seed)
    opt = get_opt(args.device, dataset=args.dataset)
    if cond_dict is None:
        if args.cond_path:
            cond_dict=np.load(args.cond_path, allow_pickle=True).item()
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))        
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = 1

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    sampling_fun = diffusion.ddim_sample_loop if args.sampler == 'ddim' else diffusion.p_sample_loop

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    model.to(dist_util.dev())
    model.eval()

    #
    # When a .txt of paired samples to Blend is provided (for Eval) ...
    if args.ref_samples and args.tgt_samples:
        dataset_npy_dir = args.dataset_npy_dir or os.path.join(opt.data_root, 'motions')
        with open(args.ref_samples, 'r') as f:
            ref_samples = [os.path.join(dataset_npy_dir, f"{line}.npy") for line in f.read().splitlines()]
        with open(args.tgt_samples, 'r') as f:
            tgt_samples = [os.path.join(dataset_npy_dir, f"{line}.npy") for line in f.read().splitlines()]
    
        _ref_dir    = os.path.dirname(os.path.abspath(args.ref_samples))
        _category   = os.path.basename(_ref_dir)                          # e.g. "walk_run"
        _bench_type = os.path.basename(os.path.dirname(_ref_dir))         # e.g. "in_skel"
        _skel_mode  = _bench_type if _bench_type in ('in_skel', 'x_skel') else 'bench'
        
        if args.intersection_only:
            _sched = args.blend_schedule
            if _sched == 'ease' and args.ease_slope != 1.0:
                _sched = f'ease_slope{args.ease_slope:g}'
            if _sched == 'static' and len(args.alpha_values) == 1:
                _a = args.alpha_values[0]
                _blend_desc = f'intersection_{_sched}_alpha{_a:.2f}'.rstrip('0').rstrip('.')
            else:
                _blend_desc = f'intersection_{_sched}'
        
        elif args.control_mode == 'ref':
            _blend_desc = 'ref'
        
        elif args.control_mode == 'tgt':
            _blend_desc = 'transfer'
        
        elif args.overlap_length > 0:
            _sched = args.blend_schedule
            if _sched == 'ease' and args.ease_slope != 1.0:
                _sched = f'ease_slope{args.ease_slope:g}'
            _blend_desc = f'temporal_{_sched}'
        
        elif args.blend_schedule == 'static' and len(args.alpha_values) == 1:
            _a = args.alpha_values[0]
            _blend_desc = f'alpha{_a:.2f}'.rstrip('0').rstrip('.')
        
        else:
            _blend_desc = args.blend_schedule
        _sampler_tag = f'{args.sampler}-sample'
        _inv_tag     = f'_ddim-inv-{diffusion.original_num_steps}{"_slerp" if args.transition_slerp else ""}' if args.ddim_inversion else ''
        out_path = pjoin(out_path, f'bench-{_skel_mode}-{_category}-{_blend_desc}_{_sampler_tag}{_inv_tag}')
    
    else:
        ref_samples = [args.ref_motion]
        tgt_samples = [args.tgt_motion]
        out_path = pjoin(out_path, 'generated')
        
    os.makedirs(out_path, exist_ok=True)
    assert len(ref_samples) == len(tgt_samples), "The number of reference and target samples must be the same"

   
    if args.control_mode == 'ref':  # for log consistency
        alpha_values = [0.0]
    elif args.control_mode == 'tgt':
        alpha_values = [1.0]
    else:
        alpha_values = args.alpha_values

    ##
    ## Main Inference Loop
    ##

    # Build completed-pair counts once from existing JSON dumps, keyed by
    # (ref_basename, tgt_basename) — e.g. ("Alligator___Walk3_22.npy", "Alligator___Bite7_8.npy").
    # Only populated when --resume is active and a pair-list is provided.
    if args.resume and args.ref_samples and args.tgt_samples:
        _purge_ghost_outputs(out_path)
        completed_pairs = _scan_completed_pairs(out_path)
    else:
        completed_pairs = {}
    expected_per_pair = len(alpha_values) * args.num_repetitions

    for pair_idx, (ref_motion, tgt_motion) in enumerate(zip(ref_samples, tgt_samples)):
        print(f"Processing pair:\nReference: {ref_motion}\nTarget: {tgt_motion}")

        if completed_pairs:
            key = (os.path.basename(ref_motion), os.path.basename(tgt_motion))
            completed = completed_pairs.get(key, 0)
            if completed >= expected_per_pair:
                print(f"  [resume] pair {pair_idx} already complete ({completed}/{expected_per_pair}), skipping.")
                continue
            elif completed > 0:
                print(f"  [resume] pair {pair_idx} partially done ({completed}/{expected_per_pair}), continuing.")
    
        # single pair at a time for now
        ref_batch, tgt_batch, filenames =  get_control_batches(ref_motion, tgt_motion, cond_dict, t5_conditioner, args, opt)
        control = format_control_batch(ref_batch, tgt_batch, args)
        if args.intersection_only:
            control = trim_control_to_intersection(control)
        batch, model_kwargs = format_gen_batch(control, cond_dict, t5_conditioner, args, opt)

        for alpha in alpha_values:
            # Format alpha tensor to perform blending
            alpha_tensor, blend_mask = format_alpha_tensor(batch, control, alpha, args)
            control.alpha = alpha_tensor.to(control.x.device)
            control.blend_mask = blend_mask.to(control.x.device)

            # in DDIM inversion we basically obtain the x_t from x_0 for each control signal
            # for such cases, as for training, the control sig. is the clean motion itself (ncontrols=1)
            if args.ddim_inversion:
                print("DDIM inversion...")
                control_cpy = deepcopy(control)
                img = rearrange(control_cpy.x, 'b s ... -> (b s) ...')
                control_cpy.x = img.unsqueeze(1)

                if isinstance(model, MoDiffAE):
                    ddim_inversion_model_kwargs = {'y': control_cpy.y, 'control': control_cpy}
                else:
                    ddim_inversion_model_kwargs = {'y': control_cpy.y}

                inverted_noise = diffusion.ddim_inversion_loop(
                    model=model,
                    img=img,
                    model_kwargs=ddim_inversion_model_kwargs,
                    device=dist_util.dev(),
                    clip_denoised=False,
                    progress=True,
                    skip_timesteps=0,
                )
                if args.control_mode == 'ref':
                    noise = inverted_noise[0].unsqueeze(0)[:, :, :, :batch.shape[-1]]
                elif args.control_mode == 'tgt':
                    noise = inverted_noise[1].unsqueeze(0)[:, :, :, :batch.shape[-1]]
                else:
                    # 'both': frame-aligned noise — pure regions get inverted noise,
                    # transition zone gets Gaussian so the model bridges freely
                    noise = build_aligned_ddim_noise(batch, control, inverted_noise[0], inverted_noise[1], alpha_tensor, args.transition_slerp)
            else:
                noise = None

            if isinstance(model, MoDiffAE):
                model_kwargs['control'] = control # attatch control
            
            # Sampling Loop
            for rep_i in range(args.num_repetitions):
                print(f'### Sampling [repetitions #{rep_i} of {args.num_repetitions}]')
                sample = sampling_fun(
                    model,
                    batch.shape,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=noise,
                    const_noise=False,
                )

                for i, motion in enumerate(sample):
                    # gather vars
                    n_joints = model_kwargs['y']["n_joints"][i].item()
                    motion = motion[:n_joints]
                    object_type = model_kwargs['y']["object_type"][i]
                    parents = model_kwargs['y']["parents"][i]
                    mean = cond_dict[object_type]['mean'][None, :]
                    std = cond_dict[object_type]['std'][None, :]
                    motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
                    offsets = cond_dict[object_type]['offsets']
                    global_positions = recover_from_bvh_ric_np(motion)
                    out_anim = None
                    #global_positions, out_anim = recover_from_bvh_rot_np(motion, parents, offsets) # uncomment this to recover XYZ pos from rots (faster, less visually appealing)
                    
                    # Name formatting
                    name_pref = f'{object_type}_rep_{rep_i}'
                    existing_npy_files = [f for f in os.listdir(out_path) if f.startswith(name_pref) and f.endswith('.npy')]
                    current_idx = len(existing_npy_files)
                    npy_name  = f"{name_pref}_#{current_idx}.npy"
                    mp4_name  = f"{name_pref}_#{current_idx}.mp4"
                    bvh_name  = f"{name_pref}_#{current_idx}.bvh"
                    json_name = f"{name_pref}_#{current_idx}.json"

                    # Save in all formats
                    if args.render_video:
                        plot_general_skeleton_3d_motion(pjoin(out_path, mp4_name), parents, global_positions, title=name_pref, fps=fps, title_wrap=20)
                    
                    np.save(pjoin(out_path, npy_name), motion) # npy motion
                    
                    if out_anim is None:
                        out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)

                    BVH.save(pjoin(out_path, bvh_name), out_anim, cond_dict[object_type]['joints_names']) # bvh motion
                    
                    _ref_s = int(control.y['crop_start_ind'][0])
                    _ref_e = _ref_s + int(control.y['lengths'][0])
                    _tgt_s = int(control.y['crop_start_ind'][1])
                    _tgt_e = _tgt_s + int(control.y['lengths'][1])
                    with open(pjoin(out_path, json_name), 'w') as f:
                        json.dump({
                            'samples': {
                                'ref':        filenames[i][0],
                                'ref_path':   ref_motion,
                                'ref_object': object_type,
                                'tgt':        filenames[i][1],
                                'tgt_path':   tgt_motion,
                                'tgt_object': char_from_npy_stem(filenames[i][1]),
                            },
                            'regions': {
                                'ref_s':         _ref_s,
                                'ref_e':         _ref_e,
                                'tgt_s':         _tgt_s,
                                'tgt_e':         _tgt_e,
                                'overlap_s':     max(_ref_s, _tgt_s),
                                'overlap_e':     min(_ref_e, _tgt_e),
                                'same_skeleton': control.y['object_type'][0] == control.y['object_type'][1],
                                'ref_n_joints':  int(control.y['n_joints'][0]),
                                'tgt_n_joints':  int(control.y['n_joints'][1]),
                            },
                            'blend': {
                                'control_mode':       args.control_mode,
                                'alpha':              alpha,
                                'blend_schedule':     args.blend_schedule,
                                'ease_slope':         args.ease_slope,
                                'overlap_length':     args.overlap_length,
                                'ref_frame_start':    _ref_s,
                                'tgt_frame_start':    _tgt_s,
                                'ref_num_loops':      args.ref_num_loops,
                                'tgt_num_loops':      args.tgt_num_loops,
                                'interpolation_mode': args.interpolation_mode,
                            },
                            'method': 'mmix',
                            'output': pjoin(out_path, npy_name),
                            'diffusion': {
                                'model_path':     args.model_path,
                                'sampler':        args.sampler,
                                'steps':          diffusion.original_num_steps,
                                'ddim_inversion': args.ddim_inversion,
                                'seed':           args.seed,
                            },
                        }, f, indent=4)
                    print(f"repetition #{rep_i} of {args.num_repetitions} ,created motion: {npy_name}")
    

def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs

def create_sample_in_batch(object_type, object_cond_dict, n_frames, max_joints, feature_len, temporal_window, t5_conditioner, motion=None, frame_start=0, loop_times=1):
    """ prepares a batch element for truebones_batch_collate() """
    parents = object_cond_dict['parents']
    n_joints = len(parents)
    mean = object_cond_dict['mean']
    std = object_cond_dict['std']

    # Repeat a given motion loop_times
    if loop_times > 1:
        n_frames *= loop_times
        if motion is not None:
            reps = [loop_times] + [1] * (motion.ndim - 1)
            motion = np.tile(motion, reps)

    # From given motion or placeholder
    if motion is not None:
        motion = (motion - mean[None]) / (std[None] + 1e-6)
        motion = np.nan_to_num(motion)
    else:
        motion = np.zeros((n_frames, n_joints, feature_len))
    assert motion.shape[0] == n_frames and motion.shape[-1] == feature_len, "sanity check"

    tpos_first_frame = object_cond_dict['tpos_first_frame']
    tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
    tpos_first_frame = np.nan_to_num(tpos_first_frame)
    joint_relations = object_cond_dict['joint_relations']
    joints_graph_dist = object_cond_dict['joints_graph_dist']
    offsets = object_cond_dict['offsets']
    joints_names_embs = encode_joints_names(object_cond_dict['joints_names'] , t5_conditioner).detach().cpu().numpy()
    kin_chain = object_cond_dict['kinematic_chains']
    temporal_mask = create_temporal_mask_for_window(temporal_window, n_frames)

    # Build the batch in correct order (check ./data_loaders/tensors.py)
    batch = [
        motion, n_frames, parents, tpos_first_frame,
        offsets, temporal_mask, joints_graph_dist,
        joint_relations, object_type, joints_names_embs,
        frame_start, mean, std, kin_chain, max_joints,
    ]
    return batch


def get_control_batches(ref_motion_path, tgt_motion_path, cond_dict, t5_conditioner, args, opt):
    ref_motion = np.load(ref_motion_path)
    tgt_motion = np.load(tgt_motion_path)
    ref_object_type = char_from_npy_stem(os.path.basename(ref_motion_path))
    tgt_object_type = char_from_npy_stem(os.path.basename(tgt_motion_path))
    ref_n_frames = ref_motion.shape[0]
    tgt_n_frames = tgt_motion.shape[0]

    ref_frame_start = args.ref_frame_start if args.overlap_length < 0 else 0
    tgt_frame_start = args.tgt_frame_start if args.overlap_length < 0 else (ref_n_frames * args.ref_num_loops) - args.overlap_length

    ref_sample = create_sample_in_batch(ref_object_type, cond_dict[ref_object_type], ref_n_frames, opt.max_joints, opt.feature_len, args.temporal_window, t5_conditioner, motion=ref_motion, frame_start=ref_frame_start, loop_times=args.ref_num_loops)
    tgt_sample = create_sample_in_batch(tgt_object_type, cond_dict[tgt_object_type], tgt_n_frames, opt.max_joints, opt.feature_len, args.temporal_window, t5_conditioner, motion=tgt_motion, frame_start=tgt_frame_start, loop_times=args.tgt_num_loops)
    
    filenames = [(os.path.basename(ref_motion_path), os.path.basename(tgt_motion_path))]
    filenames = [(f[0], f[1]) for f in filenames]

    return [ref_sample], [tgt_sample], filenames

def trim_control_to_intersection(control):
    """
    Restrict both control signals to the intersection (overlap) of their ranges.

    After this call:
      - control.x[:, 0, ..., :] holds only the ref frames that fall in [overlap_s, overlap_e)
      - control.x[:, 1, ..., :] holds only the tgt frames that fall in [overlap_s, overlap_e)
      - crop_start_ind is reset to 0 for both signals
      - lengths is set to overlap_length for both signals

    format_gen_batch will then compute gen_n_frames = overlap_length automatically.
    """
    ref_s = int(control.y['crop_start_ind'][0])
    ref_e = ref_s + int(control.y['lengths'][0])
    tgt_s = int(control.y['crop_start_ind'][1])
    tgt_e = tgt_s + int(control.y['lengths'][1])

    overlap_s = max(ref_s, tgt_s)
    overlap_e = min(ref_e, tgt_e)
    assert overlap_e > overlap_s, (
        f"Ref [{ref_s}, {ref_e}) and tgt [{tgt_s}, {tgt_e}) do not overlap — "
        "cannot use --intersection_only."
    )

    overlap_len = overlap_e - overlap_s

    # Slice each signal to the overlap portion (control.x: B x 2 x joints x feats x T)
    ref_slice = slice(overlap_s - ref_s, overlap_e - ref_s)
    tgt_slice = slice(overlap_s - tgt_s, overlap_e - tgt_s)
    control.x = torch.stack([
        control.x[:, 0, ..., ref_slice],
        control.x[:, 1, ..., tgt_slice],
    ], dim=1)

    # Reset positional metadata so both signals start at frame 0
    control.y['crop_start_ind'][0] = 0
    control.y['crop_start_ind'][1] = 0
    control.y['lengths'][0] = overlap_len
    control.y['lengths'][1] = overlap_len

    # Trim temporal mask to match the new overlap length.
    # The mask has shape (2, 1, 1, T_max+1, T_max+1); index 0 is the T-pose,
    # indices 1..T_max correspond to motion frames. Each signal's overlap
    # starts at a different offset within its original mask.
    ref_msk_s = (overlap_s - ref_s) + 1   # +1 to skip T-pose at index 0
    tgt_msk_s = (overlap_s - tgt_s) + 1
    dev = control.y['mask'].device
    ref_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                         torch.arange(ref_msk_s, ref_msk_s + overlap_len, device=dev)])
    tgt_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                         torch.arange(tgt_msk_s, tgt_msk_s + overlap_len, device=dev)])
    control.y['mask'] = torch.stack([
        control.y['mask'][0][:, :, ref_idx][:, :, :, ref_idx],
        control.y['mask'][1][:, :, tgt_idx][:, :, :, tgt_idx],
    ], dim=0)

    return control


def format_gen_batch(control, cond_dict, t5_conditioner, args, opt):

    # We generate the "REF" type (index 0)
    gen_object_type = control.y['object_type'][0]
    
    # generated animation length
    if args.control_mode == 'ref':
        gen_n_frames = control.y['lengths'][0].item() + control.y['crop_start_ind'][0].item()
    elif args.control_mode == 'tgt':
        gen_n_frames = control.y['lengths'][1].item() + control.y['crop_start_ind'][1].item()
    else:
        gen_n_frames = max(a+b for a,b in zip(control.y['lengths'], control.y['crop_start_ind'])) # both
    
    # build the gen batch
    gen_batch = create_sample_in_batch(gen_object_type, cond_dict[gen_object_type], gen_n_frames, opt.max_joints, opt.feature_len, args.temporal_window, t5_conditioner, motion=None, frame_start=0, loop_times=1)
    batch, model_kwargs = truebones_batch_collate([gen_batch])
    
    return batch, model_kwargs


def format_control_batch(ref_batch, tgt_batch, args):
    
    assert len(ref_batch) == len(tgt_batch), "Source and target batches must have the same number of samples" # sanity check
    bs = len(ref_batch)
    
    merged_batch, merged_model_kwargs = truebones_batch_collate(ref_batch + tgt_batch)
    control = ControlConfig(x=torch.zeros_like(merged_batch), y={}, alpha=torch.zeros(bs), interpolation_mode=args.interpolation_mode)

    def _format_val(v):
        """Helper to slice or interleave based on alpha mode"""
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return rearrange(v, '(s b) ... -> (b s) ...', s=2)
        return [item for pair in zip(v[:bs], v[bs:]) for item in pair]

    control.x = rearrange(merged_batch, '(s b) ... -> b s ...', s=2)
    for k, v in merged_model_kwargs['y'].items():
        control.y[k] = _format_val(v)

    control.x = control.x.to(dist_util.dev())
    return control

def build_aligned_ddim_noise(batch, control, ref_noise, tgt_noise, alpha_tensor, transition_slerp=False):
    """
    Build DDIM starting noise for the 'both' control mode.

    Pure-ref frames get the noise obtained by inverting the ref clean motion,
    pure-tgt frames (only when skeletons match) get the noise from the tgt inversion.

    Transition zone (overlap between ref and tgt ranges):
    - transition_slerp=True, same skeleton : SLERP between ref and tgt inverted noise weighted
                              by alpha_tensor, giving a smooth noise trajectory through the overlap.
    - transition_slerp=True, cross-skeleton: SLERP between ref inverted noise and the pre-drawn
                              random Gaussian noise (tgt topology is incompatible, so random noise
                              is used as the "tgt" end), giving a smooth fade from structured to free.
    - transition_slerp=False: standard Gaussian noise, leaving the model fully unconstrained.

    Frames outside both ranges always get standard Gaussian noise.

    ref_noise / tgt_noise: (max_joints, feats, T_ctrl) — output of ddim_inversion_loop,
    indexed in control space (frame 0 = first frame of that motion).
    crop_start_ind maps control-space frame 0 to output-space frame ref_s / tgt_s.
    """
    T = batch.shape[-1]
    device = ref_noise.device

    ref_s = int(control.y['crop_start_ind'][0])
    ref_e = ref_s + int(control.y['lengths'][0])
    tgt_s = int(control.y['crop_start_ind'][1])
    tgt_e = tgt_s + int(control.y['lengths'][1])
    overlap_s = max(ref_s, tgt_s)
    overlap_e = min(ref_e, tgt_e)

    same_skeleton = control.y['object_type'][0] == control.y['object_type'][1]

    # noise is initialised with pure Gaussian; _fill overwrites selected frames in-place.
    # For cross-skeleton transition SLERP we intentionally read back the Gaussian values
    # that were already drawn here, so the random tensor must be created before any fills.
    noise = torch.randn(1, *ref_noise.shape[:-1], T, device=device, dtype=ref_noise.dtype)

    def _fill(src, out_s, out_e, ctrl_offset):
        out_s, out_e = max(out_s, 0), min(out_e, T)
        if out_e <= out_s:
            return
        out_frames = torch.arange(out_s, out_e, device=device)
        noise[0, :, :, out_frames] = src[:, :, out_frames - ctrl_offset]

    # Pure-ref segments (ref range minus overlap)
    _fill(ref_noise, ref_s, min(overlap_s, ref_e), ref_s)
    _fill(ref_noise, max(overlap_e, ref_s), ref_e, ref_s)

    # Pure-tgt segments (same skeleton only — cross-skeleton stays as random Gaussian)
    if same_skeleton:
        _fill(tgt_noise, tgt_s, min(overlap_s, tgt_e), tgt_s)
        _fill(tgt_noise, max(overlap_e, tgt_s), tgt_e, tgt_s)

    # Transition zone SLERP
    if transition_slerp and overlap_e > overlap_s:
        out_s = max(overlap_s, 0)
        out_e = min(overlap_e, T)
        if out_e > out_s:
            out_frames = torch.arange(out_s, out_e, device=device)
            ref_overlap = ref_noise[:, :, out_frames - ref_s]   # (J, F, overlap_len)
            if same_skeleton:
                tgt_overlap = tgt_noise[:, :, out_frames - tgt_s]
            else:
                # Cross-skeleton: SLERP from structured ref noise toward the random Gaussian
                # that was pre-drawn for these frames (tgt topology is incompatible).
                tgt_overlap = noise[0, :, :, out_frames].clone()
            alpha_overlap = alpha_tensor.to(device)[out_frames]   # (overlap_len,)
            noise[0, :, :, out_frames] = slerp(ref_overlap, tgt_overlap, alpha_overlap)

    return noise


def build_inpainting_kwargs(batch, control):
    """
    Build inpainted_motion and inpainting_mask for the generation batch.

    Only the *pure* (non-overlapping) portions of each motion are hard-constrained:
    - Pure-ref frames (ref range minus the transition zone) → clean ref motion
    - Pure-tgt frames (tgt range minus the transition zone), only when skeletons match → clean tgt motion

    The transition/overlap zone is intentionally left free so the diffusion can blend
    smoothly without snapping at the boundary.
    """
    T = batch.shape[-1]
    device = control.x.device
    inpainted_motion = torch.zeros_like(batch, device=device)
    inpainting_mask = torch.zeros_like(batch, dtype=torch.bool, device=device)

    ref_motion = control.x[:, 0, ...]  # (bs, max_joints, feats, T_ctrl)
    tgt_motion = control.x[:, 1, ...]

    ref_s = int(control.y['crop_start_ind'][0])
    ref_e = ref_s + int(control.y['lengths'][0])
    tgt_s = int(control.y['crop_start_ind'][1])
    tgt_e = tgt_s + int(control.y['lengths'][1])

    # Overlap/transition zone — must stay free for smooth blending
    overlap_s = max(ref_s, tgt_s)
    overlap_e = min(ref_e, tgt_e)

    def _fill(motion, out_s, out_e, ctrl_offset):
        out_s, out_e = max(out_s, 0), min(out_e, T)
        if out_e <= out_s:
            return
        out_frames = torch.arange(out_s, out_e, device=device)
        inpainted_motion[:, :, :, out_frames] = motion[:, :, :, out_frames - ctrl_offset]
        inpainting_mask[:, :, :, out_frames] = True

    # Pure-ref: ref range minus overlap (two potential segments, usually only one is non-empty)
    _fill(ref_motion, ref_s, min(overlap_s, ref_e), ref_s)   # before overlap
    _fill(ref_motion, max(overlap_e, ref_s), ref_e, ref_s)   # after overlap

    # Pure-tgt (same skeleton only)
    ref_object_type = control.y['object_type'][0]
    tgt_object_type = control.y['object_type'][1]
    if ref_object_type == tgt_object_type:
        _fill(tgt_motion, tgt_s, min(overlap_s, tgt_e), tgt_s)   # before overlap
        _fill(tgt_motion, max(overlap_e, tgt_s), tgt_e, tgt_s)   # after overlap

    return inpainted_motion, inpainting_mask


def format_alpha_tensor(batch, control, alpha, args):
    T = batch.shape[-1]
    
    # Indices for reference [0] and target [1]
    ref_s, ref_e = control.y['crop_start_ind'][0], control.y['crop_start_ind'][0] + control.y['lengths'][0]
    tgt_s, tgt_e = control.y['crop_start_ind'][1], control.y['crop_start_ind'][1] + control.y['lengths'][1]
    
    # If Ref starts first, we assume 0.0 (Ref) is the "past" and 1.0 (Tgt) is the "future"
    alpha_tensor = torch.zeros((T,), device=batch.device)
    if ref_s < tgt_s:
        alpha_tensor[ref_e:] = 1.0
    else:
        alpha_tensor[:ref_s] = 1.0

    # Define the Blend Mask (The "Valid" range for conditioning)
    blend_mask = torch.zeros(T, dtype=torch.bool, device=batch.device)
    if args.control_mode in ['ref', 'both']:
        blend_mask[ref_s:ref_e] = True
    if args.control_mode in ['tgt', 'both']:
        blend_mask[tgt_s:tgt_e] = True

    if args.control_mode == 'ref':
        alpha_tensor.fill_(0.0)
    elif args.control_mode == 'tgt':
        alpha_tensor.fill_(1.0)
    else : # both
        inter_s, inter_e = max(ref_s, tgt_s), min(ref_e, tgt_e)
    
        if args.blend_schedule == 'static':
            alpha_tensor[:] = alpha
        
        elif args.blend_schedule in ['linear', 'ease'] and inter_e > inter_s:
            dist = inter_e - inter_s
            t = torch.linspace(0, 1, dist, device=batch.device)
            
            if args.blend_schedule == 'linear':
                res = t
            else: # ease — symmetric power curve, always in [0, 1]
                s = 0.5 * (1 + torch.sin(torch.pi * (t - 0.5)))  # standard ease, s∈[0,1]
                if args.ease_slope == 1.0:
                    res = s
                else:
                    res = torch.where(s <= 0.5,
                                      0.5 * (2.0 * s) ** args.ease_slope,
                                      1.0 - 0.5 * (2.0 * (1.0 - s)) ** args.ease_slope)

            # Transition from the first signal to the second
            if ref_s < tgt_s:
                alpha_tensor[inter_s:inter_e] = res
            else:
                alpha_tensor[inter_s:inter_e] = 1.0 - res

    #print(alpha_tensor.shape, blend_mask.shape, batch.shape)
    
    return alpha_tensor, blend_mask

if __name__ == "__main__":
    main()
