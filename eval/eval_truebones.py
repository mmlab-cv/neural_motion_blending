# This code is based on https://github.com/SinMDM/SinMDM/blob/main/eval/eval_mixamo.py
from eval.metrics.patched_nn import patched_nn_main
from eval.metrics.perwindow_nn import perwindow_nn, coverage
from eval.metrics.distances import avg_per_frame_dist
from tqdm import tqdm
from utils.fixseed import fixseed
import numpy as np
import torch

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
from utils.parser_util import evaluation_parser
import os
import BVH # pip install git+https://github.com/inbar-2344/Motion.git
import random
import math

def bvh2data(bvh_file_path):
    # extract 6D representation from BVH file
    anim, joint_names, dt = BVH.load(bvh_file_path)
    anim_rot = torch.from_numpy(anim.rotations.rotation_matrix(cont6d=True))  # [n_frames, n_joints, 6]
    return anim_rot.view(anim_rot.shape[0], -1)  # [n_frames, n_joints*6]

def npy2data(npy_file_path, mode):
    # assuming unnormalized data!
    try:
        anim = np.load(npy_file_path)
    except:
        # npys in SinMDM format
        anim = np.load(npy_file_path, allow_pickle=True).item()
        anim = anim['motion_raw'].transpose(0, 3, 1, 2)  # n_samples x n_joints x n_feats x n_frames  ==>  n_samples x n_frames x n_joints x n_feats
    y_root = anim[..., 0, 1]
    anim = anim[..., 1:, :]  # [1:] means excluding the root # [n_frames, n_joints, (3 ric pos, 6 rot, 3 linear vel, 1 foot contact)] 
    if mode == 'rot':
        anim_out = torch.from_numpy(anim[..., 3:9]).reshape(anim.shape[:-2]+(-1,))
    elif mode == 'loc':
        anim_out = torch.from_numpy(anim[..., :3])
        anim_out[..., 1] = anim_out[..., 1] - y_root[..., np.newaxis]  # subtract y_root from y of each joint, as they are all absolute
        anim_out = anim_out.reshape(anim.shape[:-2]+(-1,))
    else:
        raise ValueError(f'Invalid mode [{mode}]')
    
    if anim_out.dtype == torch.float32:  # sinmdm returns float32 tensors. npy_rot crashes on float32.
        anim_out = anim_out.double()
    return anim_out

def character2data(dir_path, character_name, mode):
    # collect all BVH files in the character directory
    file_ext = '.' + mode.split('_')[0]
    bvh_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext) and f.split('_')[0]==character_name]
    if len(bvh_files) == 0:
        print(f'No BVH [{character_name}] files found in [{dir_path}]')
        return None
    #assert len(bvh_files) > 0, f'No BVH [{character_name}] files found in [{dir_path}]'

    if file_ext == '.bvh':
        extract_fn = bvh2data
    elif file_ext == '.npy':
        extract_fn = lambda path: npy2data(path, mode=mode.split('_')[1])
    else:
        raise ValueError(f'Invalid file_ext [{file_ext}]')

    data = []
    for bvh_file in bvh_files:
        data_sample = extract_fn(bvh_file)
        if data_sample.shape[-2] > 20:
            if data_sample.dim() == 2:
                data_sample = [data_sample]
            else:
                data_sample = [motion for motion in data_sample]
            data.extend(data_sample)
        # print(data[-1].shape)

    print(f' [{character_name}] Found {len(data)} motions with length > 20')
    return data    

def concat_list_of_tensors(tensors):
    return torch.cat(tensors, dim=0)

def print_results(character_name, char_eval_dict):
    print(f'[{character_name}] RESULTS:')
    print('='*10)
    for metric in char_eval_dict:
        print(f'[{metric}] {char_eval_dict[metric]["mean"]:.2f}±{char_eval_dict[metric]["std"]:.2f}')
    print('='*10)

def evaluate_character(args, character_name):

    gen_samples = character2data(args.eval_gen_dir, character_name, args.eval_mode)
    if gen_samples is None:
        return None
    gt_samples = character2data(args.eval_gt_dir, character_name, args.eval_mode)

    n_gen_samples = len(gen_samples)
    n_gen_frames = sum([e.shape[0] for e in gen_samples])
    n_gt_samples = len(gt_samples)
    n_gt_frames = sum([e.shape[0] for e in gt_samples])
    n_single_gen_frames = 120  # Hardcoded
    assert n_gt_frames*2 <= n_gen_frames  # all benchmark examples must apply to this
    assert all([e.shape[0] == n_single_gen_frames for e in gen_samples])

    global_variations = []
    local_variations = []
    coverages = []
    inter_div_dist = []
    intra_div_dist = []
    intra_div_gt_diff = []

    tmin = 15
    n_repetitions = 10 # for random window/sample metrics
    # sample_size = min(10, n_gen_samples) # for random subset metrics
    sample_size = n_gt_frames // n_single_gen_frames + 1
    coverage_threshold = math.radians(20)  # 20 degrees

    eval_type = args.eval_mode.split('_')[1]
    use_pos = eval_type == 'loc'

    # Calc GT diversity
    gt_intra_div_dist = []
    for i in range(n_gt_samples):
        n_frames = gt_samples[i].shape[0]
        gt_offsets = np.random.randint(n_frames-tmin, size=(n_repetitions, 2))
        for j in range(n_repetitions):
            gt_intra_div_dist.append(avg_per_frame_dist(gt_samples[i][gt_offsets[j,0]:gt_offsets[j,0]+tmin], 
                                                        gt_samples[i][gt_offsets[j,1]:gt_offsets[j,1]+tmin], norm=eval_type))
    gt_intra_div_dist_mean = np.mean(gt_intra_div_dist)

    for i in range(n_gen_samples):
        n_frames = gen_samples[i].shape[0]
        offsets = np.random.randint(n_frames-tmin, size=(2))
        local_variations.append(perwindow_nn(gen_samples[i], concat_list_of_tensors(gt_samples), tmin=tmin, use_pos=use_pos))
        inter_div_dist.append(avg_per_frame_dist(gen_samples[i], gen_samples[i-1], norm=eval_type))
        intra_div_dist.append(avg_per_frame_dist(gen_samples[i][offsets[0]:offsets[0]+tmin], 
                                                 gen_samples[i][offsets[1]:offsets[1]+tmin], norm=eval_type))
        intra_div_gt_diff.append(abs(intra_div_dist[-1]-gt_intra_div_dist_mean))

        # For each gen: run compared to each GT and take the min
        all_global_variations = []
        for j in range(n_gt_samples):
            all_global_variations.append(patched_nn_main(gen_samples[i], gt_samples[j], use_pos=use_pos))
        global_variations.append(min(all_global_variations))
        
    for j in range(n_repetitions):
        unique_gen_subset = random.sample(gen_samples, sample_size)
        coverages.append(coverage(concat_list_of_tensors(unique_gen_subset), gt_samples, threshold=coverage_threshold, tmin=tmin, use_pos=use_pos))
       
    char_eval_dict = {
        'coverage': {'mean': np.mean(coverages) * 100, 'std': np.std(coverages) * 100},
        'global_diversity': {'mean': np.mean(global_variations), 'std': np.std(global_variations)},
        'local_diversity': {'mean': np.mean(local_variations), 'std': np.std(local_variations)},
        'inter_diversity_dist': {'mean': np.mean(inter_div_dist), 'std': np.std(inter_div_dist)},
        'intra_diversity_dist': {'mean': np.mean(intra_div_dist), 'std': np.std(intra_div_dist)},
        'gt_intra_diversity_dist': {'mean': np.mean(gt_intra_div_dist), 'std': np.std(gt_intra_div_dist)},
        'intra_div_gt_diff': {'mean': np.mean(intra_div_gt_diff), 'std': np.std(intra_div_gt_diff)},
    }
    print_results(character_name, char_eval_dict)
    return char_eval_dict


def eval_full_benchmark(args):
    file_ext = '.' + args.eval_mode.split('_')[0]
    
    if args.benchmark_path == '':
        all_bvh_files = [f for f in os.listdir(args.eval_gt_dir) if f.endswith(file_ext)]
        all_characters = list(set([f.split('_')[0] for f in all_bvh_files]))
        excluded_characters = args.characters_to_exclude.split(',')
        evaluated_characters = [c for c in all_characters if c not in excluded_characters]
    else:
        with open(args.benchmark_path, 'r') as fr:
            evaluated_characters = [c.strip() for c in fr.readlines()]

    print(f'Evaluating [{len(evaluated_characters)}] characters:')

    characters_eval_dict = {}
    skip_characters = set()
    for char in evaluated_characters:
        eval_dict = evaluate_character(args, char)
        if eval_dict is None:
            print(f'WARNING!!!! Wasnt able to evaluate [{char}]')
            skip_characters.add(char)
            continue
        characters_eval_dict[char] = eval_dict
    
    # Avraged results
    metrics = list(list(characters_eval_dict.values())[0].keys())
    characters_eval_dict['FINAL'] = {}
    for m in metrics:
        mean = np.mean([characters_eval_dict[c][m]['mean'] for c in evaluated_characters if c not in skip_characters])
        std = np.std([characters_eval_dict[c][m]['mean'] for c in evaluated_characters if c not in skip_characters])
        characters_eval_dict['FINAL'][m] = {'mean': mean, 'std': std}

    print_results('FINAL', characters_eval_dict['FINAL'])
    return characters_eval_dict


if __name__ == "__main__":

    args = evaluation_parser()
    fixseed(args.seed)
    print(f'EVAL MODE IS [{args.eval_mode}]')

    assert os.path.isdir(args.eval_gen_dir), f'Invalid gen dir [{args.eval_gen_dir}]'
    assert os.path.isdir(args.eval_gt_dir), f'Invalid gt dir [{args.eval_gt_dir}]'

    log_file = os.path.join(os.path.dirname(args.eval_gen_dir),
                            'eval_' + os.path.basename(args.eval_gen_dir) + '_mode_' + args.eval_mode + args.unique_str + '.log')
    print(f'Will save to log file [{log_file}]')

    eval_dict = eval_full_benchmark(args)

    with open(log_file, 'w') as fw:
        fw.write(str(eval_dict))
    np.save(log_file.replace('.log', '.npy'), eval_dict)