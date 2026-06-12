"""
Build ref.txt / tgt.txt for the Mixamo retargeting benchmark.

For each (src_char, tgt_char) pair and each shared test-set action stem:
  ref line = tgt_char npy stem  (target skeleton — generation scaffold)
  tgt line = src_char npy stem  (source motion  — style to transfer)

mix.py is then called with --control_mode tgt, which generates on the
ref (= tgt_char) skeleton using the tgt (= src_char) motion as style.

Two sub-benchmarks are always written:
  <out_dir>/isomo/ref.txt  + tgt.txt  — ISO characters only
  <out_dir>/homeo/ref.txt  + tgt.txt  — HOMEO characters only

Usage
-----
python eval/benchmarks/retargeting/build_mixamo_retargeting_bench.py \
    [--dataset_dir dataset/mixamo_processed] \
    [--out_dir eval/benchmarks/retargeting/mixamo_retargeting]
"""
import argparse
import itertools
import os
import numpy as np
from os.path import join as pjoin

# MoMa validation characters (dataloader_mixamo.py, mode='validation'):
#   iso:   Aj, BigVegas, Kaya, SportyGranny
#   homeo: Goblin_m, Mousey_m, Mremireh_m, Vampire_m
ISO   = ['Aj', 'BigVegas', 'Kaya', 'SportyGranny']
HOMEO = ['Goblin_m', 'Mousey_m', 'Mremireh_m', 'Vampire_m']


def get_stems(char, filelist_dir):
    # _m chars share the filelist with their base character (e.g. Goblin_m -> test_goblin.txt)
    base = char[:-2] if char.endswith('_m') else char
    flist = pjoin(filelist_dir, f'test_{base.lower()}.txt')
    if not os.path.isfile(flist):
        return set()
    with open(flist) as f:
        return {os.path.splitext(l.strip())[0] for l in f if l.strip()}


def find_npy_stems(stem, char, motions_dir):
    """Return all segment stems for (char, action stem), sorted by trailing counter."""
    prefix = char + '_'
    avoid  = char + '_m_' if not char.endswith('_m') else None
    matches = []
    for fname in os.listdir(motions_dir):
        if not fname.endswith('.npy') or not fname.startswith(prefix):
            continue
        if avoid and fname.startswith(avoid):
            continue
        body = fname[len(prefix):-4]
        parts = body.rsplit('_', 1)
        if len(parts) == 2 and parts[0] == stem and parts[1].isdigit():
            matches.append((int(parts[1]), fname[:-4]))
    matches.sort()
    return [stem_name for _, stem_name in matches]


def build_bench(chars, motions_dir, filelist_dir, out_dir):
    chars = [c for c in chars
             if os.path.isfile(pjoin(filelist_dir,
                               f'test_{(c[:-2] if c.endswith("_m") else c).lower()}.txt'))]

    stems_by_char = {c: get_stems(c, filelist_dir) for c in chars}
    for c, s in stems_by_char.items():
        print(f'  {c}: {len(s)} test stems')

    ref_lines, tgt_lines = [], []
    pairs = list(itertools.permutations(chars, 2))
    skipped_missing = 0
    skipped_mismatch = 0
    for src, tgt in pairs:
        shared = sorted(stems_by_char[src] & stems_by_char[tgt])
        for stem in shared:
            src_segs = find_npy_stems(stem, src, motions_dir)
            tgt_segs = find_npy_stems(stem, tgt, motions_dir)
            if not src_segs or not tgt_segs:
                skipped_missing += 1
                continue
            src_frames = [np.load(pjoin(motions_dir, s + '.npy')).shape[0] for s in src_segs]
            tgt_frames = [np.load(pjoin(motions_dir, s + '.npy')).shape[0] for s in tgt_segs]
            if src_frames != tgt_frames:
                skipped_mismatch += 1
                continue
            for src_npy, tgt_npy in zip(src_segs, tgt_segs):
                ref_lines.append(tgt_npy)
                tgt_lines.append(src_npy)

    os.makedirs(out_dir, exist_ok=True)
    ref_path = pjoin(out_dir, 'ref.txt')
    tgt_path = pjoin(out_dir, 'tgt.txt')
    with open(ref_path, 'w') as f:
        f.write('\n'.join(ref_lines))
    with open(tgt_path, 'w') as f:
        f.write('\n'.join(tgt_lines))

    if skipped_mismatch:
        print(f'  [WARNING] {skipped_mismatch} stems skipped due to frame-count mismatch across characters (different source BVH lengths)')
    print(f'  -> {len(ref_lines)} pairs ({skipped_missing} missing) written to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='dataset/mixamo_processed')
    parser.add_argument('--out_dir',
                        default='eval/benchmarks/retargeting/mixamo_retargeting')
    args = parser.parse_args()

    motions_dir  = pjoin(args.dataset_dir, 'motions')
    filelist_dir = pjoin(args.dataset_dir, 'filelist', 'test')

    print('--- ISO ---')
    build_bench(ISO,   motions_dir, filelist_dir, pjoin(args.out_dir, 'isomo'))
    print('--- HOMEO ---')
    build_bench(HOMEO, motions_dir, filelist_dir, pjoin(args.out_dir, 'homeo'))


if __name__ == '__main__':
    main()
