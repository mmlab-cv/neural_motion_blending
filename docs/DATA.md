# 🦴 Data

The model trains and evaluates on the **Truebones Zoo** collection. Whatever the source, every
processed dataset ends up in the same shape: `motions/*.npy` (per-frame motion features),
`bvhs/*.bvh` (normalized BVHs), `animations/*.mp4` (sanity-check renders), and a `cond.npy` file
holding each character's skeleton conditioning (mean/std, joint parents, offsets, joint names, ...).

📁 `assets/truebones/` ships ready-to-use sample clips (`.npy` + matching `.bvh`, one per character)
precisely so you can try sampling/blending (see [SAMPLE.md](SAMPLE.md)) without buying or processing
anything first.

## 🐾 Truebones Zoo

Truebones Zoo is a closed-source, paid BVH collection — buy it from
[truebones.gumroad.com/l/skZMC](https://truebones.gumroad.com/l/skZMC), then extract the raw BVHs to
`dataset/truebones/zoo/Truebone_Z-OO/` (one subdirectory per character) and process it locally:

```bash
python -m utils.create_truebones_dataset
```

This writes the processed set to `dataset/truebones/zoo/truebones_processed/`.

## 🧩 Adding your own skeleton

The model isn't limited to characters that shipped in Truebones — `utils/process_new_skeleton.py`
is the tool for bringing in **any** new character. Point it at a folder of BVH files for that
character and it runs the same normalization/feature-extraction pipeline used for Truebones,
producing a `cond.npy` the model can condition on:

```bash
python -m utils.process_new_skeleton \
    --object_name   Dog \
    --bvh_dir       path/to/bvh_folder \
    --save_dir      path/to/output_dir \
    --face_joints_names RLeg1 LLeg1 RArm1 LArm1
```

`--face_joints_names` takes the [right hip, left hip, right shoulder, left shoulder] joint names (or
whatever the closest equivalent is on your rig) — they're used to orient the skeleton consistently.
More BVH files in `--bvh_dir` means more accurate normalization stats; you can optionally point
`--tpos_bvh` at a dedicated rest-pose file if one exists.

The output has the same `motions/bvhs/animations/cond.npy` layout as Truebones. To actually generate
or blend motion with the new skeleton, hand its `cond.npy` to `sample/mix.py --cond_path` — see
[SAMPLE.md](SAMPLE.md). This is how you'd register a character that was never part of Truebones at
all, not just reprocess an existing one.

## 🔤 T5 joint-name cache (recommended)

```bash
python -m utils.build_t5_cache --dataset truebones
```

Joint names are embedded with a T5 model at load time; this precomputes and caches those embeddings
(`t5_cache.npz`) so training/sampling don't reload T5 on every run.
