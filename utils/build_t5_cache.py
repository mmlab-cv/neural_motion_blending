"""
Build and save a T5 joint-name embedding cache for a processed dataset.

Collects every unique joint name across all characters in cond.npy, encodes
each one exactly once, and writes t5_cache.npz next to cond.npy.
The npz stores one array per unique joint name (key = joint name string,
shape (768,) for t5-base), so characters sharing joint names reuse the same
embedding at load time.

MotionDataset loads this file at startup instead of re-running T5 every time.

Run from the mmix root:
    python -m utils.build_t5_cache --dataset mixamo
    python -m utils.build_t5_cache --dataset truebones
"""

import argparse
import os
import numpy as np
import torch
from os.path import join as pjoin

from data_loaders.truebones.truebones_utils.get_opt import DATASET_DIRS
from model.modules.conditioners import T5Conditioner

T5_NAME    = 't5-base'
CACHE_FILE = 't5_cache.npz'


def build_t5_cache(dataset: str, t5_name: str = T5_NAME):
    data_root  = DATASET_DIRS[dataset]
    cond_path  = pjoin(data_root, 'cond.npy')
    cache_path = pjoin(data_root, CACHE_FILE)

    if not os.path.isfile(cond_path):
        raise FileNotFoundError(f"cond.npy not found at {cond_path}")

    cond_dict = np.load(cond_path, allow_pickle=True).item()

    unique_names = sorted({
        name
        for entry in cond_dict.values()
        for name in entry['joints_names']
        if name is not None
    })
    print(f"Dataset: {dataset}  |  characters: {len(cond_dict)}  |  unique joint names: {len(unique_names)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t5 = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device=device)

    tokens = t5.tokenize(unique_names)
    embs   = t5(tokens).detach().cpu().numpy()   # (N_unique, 768)

    arrays = {name: embs[i] for i, name in enumerate(unique_names)}
    np.savez(cache_path, **arrays)
    print(f"Saved {len(arrays)} embeddings -> {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASET_DIRS.keys()), default='mixamo')
    parser.add_argument('--t5_name', default=T5_NAME)
    args = parser.parse_args()
    build_t5_cache(args.dataset, args.t5_name)


if __name__ == '__main__':
    main()
