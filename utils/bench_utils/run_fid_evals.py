"""
run_fid_evals.py — Batch launcher for fid_truebones_blend evaluations.

Fixed across all runs:
  --eval_gt_dir   dataset/truebones/zoo/truebones_processed/
  --gen_frames    zones
  --gt_window_size 2.0

Each entry in EVAL_DIRS is a 3-tuple:
  (eval_gen_dir, source, gt_mode)

Results are saved as JSON next to the eval_gen_dir. Already-completed runs
(JSON exists and is non-empty) are skipped unless --force is passed.

Usage:
  python utils/bench_utils/run_fid_evals.py [--force] [--dry_run] [--filter SUBSTR]
"""

import argparse
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GT_DIR         = 'dataset/truebones/zoo/truebones_processed/'
GEN_FRAMES     = 'transition'
GT_WINDOW_SIZE = 2.0
MODEL_PATH     = 'save/modiffae_truebones_all_globpool/model000449998.pt'

# (eval_gen_dir, source, gt_mode)
EVAL_DIRS = [ # In-Skel Walk_Run Temporal
    ('save/nla_blend/bench-in_skel-walk_run-temporal_linear', 'bvh', 'both'),
    ('save/all_model_dataset_truebones_bs_16_latentdim_128/samples_all_model_dataset_truebones_bs_16_latentdim_128_000459999_seed10/bench-in_skel-walk_run-temporal_linear_ddim-sample_ddim-inv-100_slerp', 'npy', 'both'),
    ('save/modiffae_truebones_all_globpool/samples_modiffae_truebones_all_globpool_000449998_seed10/bench-in_skel-walk_run-temporal_linear_ddim-sample_ddim-inv-100_slerp', 'npy', 'both'),
    ('save/modiffae_truebones_all_globpool/samples_modiffae_truebones_all_globpool_000449998_seed10/bench-in_skel-walk_run-temporal_linear_ddpm-sample', 'npy', 'both'),
    ('save/nla_blend/bench-in_skel-walk_run-temporal_ease_slope2', 'bvh', 'both'),
    ('save/all_model_dataset_truebones_bs_16_latentdim_128/samples_all_model_dataset_truebones_bs_16_latentdim_128_000459999_seed10/bench-in_skel-walk_run-temporal_ease_slope2_ddim-sample_ddim-inv-100_slerp', 'npy', 'both'),
    ('save/modiffae_truebones_all_globpool/samples_modiffae_truebones_all_globpool_000449998_seed10/bench-in_skel-walk_run-temporal_ease_slope2_ddim-sample_ddim-inv-100_slerp', 'npy', 'both'),
    ('save/modiffae_truebones_all_globpool/samples_modiffae_truebones_all_globpool_000449998_seed10/bench-in_skel-walk_run-temporal_ease_slope2_ddpm-sample', 'npy', 'both'),

]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def result_path(eval_gen_dir: str, gt_mode: str) -> str:
    return f'{eval_gen_dir.rstrip("/")}__fid_zones__{gt_mode}.json'


def run_eval(eval_gen_dir: str, source: str, gt_mode: str,
             force: bool, dry_run: bool) -> bool:
    out_json = result_path(eval_gen_dir, gt_mode)

    if not force and os.path.exists(out_json):
        try:
            with open(out_json) as f:
                data = json.load(f)
            if data:
                print(f'[SKIP] {os.path.basename(eval_gen_dir)}  gt_mode={gt_mode}')
                return True
        except Exception:
            pass  # corrupted → rerun

    cmd = [
        sys.executable, '-m', 'eval.fid_truebones_blend',
        '--eval_gen_dir',    eval_gen_dir,
        '--eval_gt_dir',     GT_DIR,
        '--model_path',      MODEL_PATH,
        '--source',          source,
        '--gen_frames',      GEN_FRAMES,
        '--gt_mode',         gt_mode,
        '--gt_window_size',  str(GT_WINDOW_SIZE),
        '--output_json',     out_json,
    ]

    print(f'\n{"="*70}')
    print(f'  DIR    : {eval_gen_dir}')
    print(f'  source={source}  gt_mode={gt_mode}  gt_window_size={GT_WINDOW_SIZE}')
    print(f'  OUT    : {out_json}')
    print(f'  CMD    : {" ".join(cmd)}')

    if dry_run:
        print('  [DRY RUN — not executed]')
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--force',   action='store_true',
                   help='Re-run even if output JSON already exists.')
    p.add_argument('--dry_run', action='store_true',
                   help='Print commands without running them.')
    p.add_argument('--filter',  default='',
                   help='Only run eval_gen_dirs containing this substring.')
    args = p.parse_args()

    total, skipped, failed = 0, 0, 0
    for eval_gen_dir, source, gt_mode in EVAL_DIRS:
        if args.filter and args.filter not in eval_gen_dir:
            continue
        if not os.path.isdir(eval_gen_dir):
            print(f'[WARN] directory not found: {eval_gen_dir} — skipping.')
            continue
        total += 1
        out_json = result_path(eval_gen_dir, gt_mode)
        if not args.force and os.path.exists(out_json):
            try:
                if json.load(open(out_json)):
                    skipped += 1
                    print(f'[SKIP] {os.path.basename(eval_gen_dir)}  gt_mode={gt_mode}')
                    continue
            except Exception:
                pass
        ok = run_eval(eval_gen_dir, source, gt_mode, force=args.force, dry_run=args.dry_run)
        if not ok:
            failed += 1

    print(f'\n{"="*70}')
    print(f'Done. total={total}, skipped={skipped}, failed={failed}')


if __name__ == '__main__':
    main()
