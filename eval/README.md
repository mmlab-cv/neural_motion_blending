# 📊 Evaluation

Assumes the Truebones dataset is already processed (see [docs/DATA.md](../docs/DATA.md)).

## 🧮 Motion-blending quality (`eval_truebones_blend.py`)

Evaluates `sample/mix.py` / `sample/nla_blend.py` outputs against GT — fidelity, diversity and
smoothness of each blend zone (pre / transition / post):

```bash
python -m eval.eval_truebones_blend \
    --eval_gen_dir <path/to/mix_output> \
    --eval_gt_dir  dataset/truebones/zoo/truebones_processed/bvhs \
    --source bvh --feats loc --root_relative
```

Results (mean/std/median per metric, split into `ALL`/`IN_SKEL`/`X_SKEL`) are written next to
`--eval_gen_dir`. Pass `--ref_samples`/`--tgt_samples` (the same pair-list files used to generate
the blends) to prioritize relevant GT clips.

## 📈 Latent-space FID (`fid_truebones_blend.py`)

FID between generated and GT motion in the MoDiffAE semantic latent space:

```bash
python -m eval.fid_truebones_blend \
    --eval_gen_dir <path/to/mix_output> \
    --eval_gt_dir  dataset/truebones/zoo/truebones_processed \
    --model_path   save/truebones_globpool/model000449998.pt \
    --source npy --gen_frames transition --gt_mode both --gt_window_size 2.0
```

## 🎯 Benchmark pairs

Use the `ref.txt`/`tgt.txt` clip-pair lists already provided in `eval/benchmarks/blend/` — e.g.
`eval/benchmarks/blend/in_skel/walk_run/` (same-character pairs) or
`eval/benchmarks/blend/x_skel/walk_run/` (cross-character pairs). Pass these directly to
`sample/mix.py` via `--ref_samples`/`--tgt_samples`, and to `eval_truebones_blend.py` /
`fid_truebones_blend.py` the same way.