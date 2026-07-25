# 🎬 Sampling & Motion Blending

`sample/mix.py` generates a motion on the **reference** skeleton, conditioned on a **reference**
motion and/or a **target** motion (same or different skeleton). It covers two distinct use cases,
selected via `--control_mode` — each with its own recommended sampler setup:

- 🌗 **Blending** (`--control_mode both`) — smoothly transition from the ref motion into the tgt
  motion. Default/recommended setup: **DDIM sampling with DDIM inversion** (`--sampler ddim
  --ddim_inversion`), which anchors the pure ref/tgt zones to their source clips and only lets the
  model improvise in the transition — much more faithful than plain DDPM.
- 🔁 **Retargeting** (`--control_mode tgt`) — pure motion transfer: the tgt motion's *content* is
  generated onto the ref skeleton, ignoring the ref clip's own motion entirely. Default/recommended
  setup: **DDPM** (`--sampler ddpm`) — there's no transition zone to anchor, so plain stochastic
  sampling is used. Since only the ref clip's *skeleton identity* matters here (its motion content
  is discarded), `--ref_motion` can point at **any** clip belonging to the desired creature — you're
  effectively just picking a skeleton by name.

The examples below use the sample clips bundled in `assets/truebones/` — pre-processed `.npy`/`.bvh`
pairs, one clip per character — so you can try this without owning the full Truebones dataset. Swap
in your own paths once you have it processed. `truebones_globpool` is the checkpoint matching the
model reported in the paper (see [TRAIN.md](TRAIN.md#-modiffae--the-papers-model)); the
attention-pool variant `truebones_attnpool` also works as a drop-in `--model_path`.

## 🌗 Blending

Crab attack blended into a Coyote attack:

```bash
python -m sample.mix \
    --model_path save/truebones_globpool/model000449998.pt \
    --ref_motion assets/truebones/Crab___Attack3_234.npy \
    --tgt_motion assets/truebones/Coyote___Attack3_224.npy \
    --control_mode both --blend_schedule ease --overlap_length 30 \
    --sampler ddim --ddim_inversion --transition_slerp
```

`--blend_schedule {static,linear,ease}` shapes the alpha ramp across the overlap
(`--alpha_values`, `--ease_slope`); `--overlap_length N` sets how many frames ref/tgt share.
`--transition_slerp` smoothly interpolates the inverted noise itself through the transition zone
(same-skeleton pairs only — otherwise it falls back to Gaussian noise there).

## 🔁 Retargeting

The Flamingo's motion, retargeted onto the Scorpion skeleton:

```bash
python -m sample.mix \
    --model_path save/truebones_globpool/model000449998.pt \
    --ref_motion assets/truebones/Scorpion___SlowForward_839.npy \
    --tgt_motion assets/truebones/Flamingo_Flamingo_OneLEgBEnt_353.npy \
    --control_mode tgt --sampler ddpm --num_repetitions 5
```

`--ref_motion` here just names the Scorpion skeleton — any other Scorpion clip would generate the
identical retarget target. `--num_repetitions` draws multiple independent DDPM samples since there's
no deterministic inversion anchoring the output.

### 🩻 Retargeting onto a skeleton with no motion clip on disk

`--ref_motion` only needs to name a skeleton, so it doesn't need a real clip — any character already
in your processed `cond.npy` works, even with zero BVH/npy files for it locally.
`utils/build_tpose_ref.py` builds a throwaway placeholder from that character's T-pose:

```bash
python -m utils.build_tpose_ref --list                    # which characters cond.npy covers
python -m utils.build_tpose_ref --object_type Alligator    # writes ./Alligator_tpose.npy
```

```bash
python -m sample.mix \
    --model_path save/truebones_globpool/model000449998.pt \
    --ref_motion Alligator_tpose.npy \
    --tgt_motion assets/truebones/Flamingo_Flamingo_OneLEgBEnt_353.npy \
    --control_mode tgt --sampler ddpm
```

## ⚙️ Common flags

| Flag | Effect |
|---|---|
| `--cond_path` | Use a skeleton outside Truebones (see [DATA.md](DATA.md#-adding-your-own-skeleton)) |
| `--ref_samples`/`--tgt_samples` | Batch mode: text files of clip pairs (one per line), used for benchmarks |
| `--render_video False` | Skip the `.mp4` stick-figure render |

Model/dataset args are loaded from the checkpoint's `args.json` — only sampling/blend flags need to
be passed. Outputs (`.npy`, `.bvh`, `.mp4`, `.json` metadata) go to `<output_dir>/generated/` next to
the checkpoint; the JSON records the blend region boundaries and every parameter used.

## 🧵 `sample/nla_blend.py` — classical baseline

A non-learned NLA crossfade baseline, run inside Blender (requires `Motion` importable from
Blender's Python):

```bash
blender --background --python sample/nla_blend.py -- \
    --ref_motion dataset/truebones/zoo/truebones_processed/bvhs/Ostrich___Walk_591.bvh \
    --tgt_motion dataset/truebones/zoo/truebones_processed/bvhs/Ostrich___Attack_581.bvh \
    --overlap_length 30 --blend_schedule ease --output_dir save/nla_blend
```

Takes BVH inputs and only supports same-skeleton blends (ref/tgt must be the same character), so it
can't be demoed with the bundled `assets/truebones/` clips — each is a different character. Point it
at two clips of the same character from your own processed dataset instead. Outputs a `.bvh` +
`.json` pair using the same schema as `mix.py`, so both feed the same eval scripts (see
[eval/README.md](../eval/README.md)).
