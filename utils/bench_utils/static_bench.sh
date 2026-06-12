# Static (intersection-only) benchmark: both controls start at frame 0 and are blended
# at a fixed alpha=0.5. Only the overlapping frames [0, min(ref_len, tgt_len)) are evaluated.
# This is the "static" counterpart to the temporal/ease benchmark.
#
# mix.py output dir suffix:  bench-$TYPE-$BENCH-intersection_static_alpha0.5_<sampler>...
# nla_blend.py output dir:   bench-$TYPE-$BENCH-intersection_static_alpha0.5

# Usage: bash utils/bench_utils/static_bench.sh <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [eval_only]
#   BENCH      : benchmark category, e.g. walk_run                     (required)
#   TYPE       : in_skel or x_skel                                     (default: in_skel)
#   SECTIONS   : quoted space-separated section numbers to run         (default: "" = all)
#                e.g. "2 3" runs only sections 2 and 3
#   MODEL_NAME : model directory name under ./save/                    (default: modiffae_truebones_all)
#   CKPT       : checkpoint iter, e.g. 000599998                       (default: 000599998)
#   eval_only  : pass "eval_only" to skip generation                   (default: "")
#
# Sections:
#   1  NLA baseline
#   2  AnyTop + DDIM inversion + SLERP
#   3  MoDiffAE + DDIM inversion + SLERP
#   4  MoDiffAE + DDIM inversion (no SLERP)
#   5  MoDiffAE + DDPM (5 repetitions)
BENCH=${1:?"Usage: $0 <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [eval_only]  — e.g. $0 walk_run in_skel"}
TYPE=${2:-in_skel}
SECTIONS=${3:-}
# Current default model: modiffae_truebones_all / 000599998
MODEL_NAME=${4:-modiffae_truebones_all}
CKPT=${5:-000599998}
EVAL_ONLY=${6:-}

SCHEDULE=static
ALPHA=0.5

REF_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/ref.txt
TGT_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/tgt.txt
GT_DIR=./dataset/truebones/zoo/truebones_processed/bvhs/

# ── Model versions ────────────────────────────────────────────────────────────
MODEL_PATH=./save/$MODEL_NAME/model${CKPT}.pt
MODEL_OUT_BASE=./save/$MODEL_NAME/samples_${MODEL_NAME}_${CKPT}_seed10

# AnyTop baseline (fixed checkpoint)
ANYTOP_NAME=all_model_dataset_truebones_bs_16_latentdim_128
ANYTOP_CKPT=000459999
ANYTOP_PATH=./save/$ANYTOP_NAME/model${ANYTOP_CKPT}.pt
ANYTOP_OUT_BASE=./save/$ANYTOP_NAME/samples_${ANYTOP_NAME}_${ANYTOP_CKPT}_seed10
# ─────────────────────────────────────────────────────────────────────────────

BENCH_OUT=./bench/static-$TYPE-$BENCH
EVAL_SUFFIX="__source_bvh__feats_loc__rootrel"
mkdir -p $BENCH_OUT

# Returns 0 (true) if section N should run: either SECTIONS is empty (run all)
# or N appears in the SECTIONS list.
_sec() { [ -z "$SECTIONS" ] || echo " $SECTIONS " | grep -qw "$1"; }

# ── Section 1: NLA baseline ───────────────────────────────────────────────────
# Output dir: ./save/nla_blend/bench-$TYPE-$BENCH-intersection_static_alpha0.5
if _sec 1; then
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        blender --background --python ./sample/nla_blend.py -- --blend_schedule $SCHEDULE --alpha_values $ALPHA --ref_samples $REF_LIST --tgt_samples $TGT_LIST --overlap_only
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir ./save/nla_blend/bench-$TYPE-$BENCH-intersection_static_alpha0.5 --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
    cp ./save/nla_blend/bench-$TYPE-$BENCH-intersection_static_alpha0.5${EVAL_SUFFIX}.json $BENCH_OUT/nla.json
fi

# ── Section 2: AnyTop + DDIM inversion + SLERP ───────────────────────────────
# Output: $ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp
if _sec 2; then
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $ANYTOP_PATH --num_repetitions 1 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --blend_schedule $SCHEDULE --alpha_values $ALPHA --intersection_only --sampler ddim --ddim_inversion --transition_slerp
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
    cp $ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp${EVAL_SUFFIX}.json $BENCH_OUT/${ANYTOP_NAME}_${ANYTOP_CKPT}_ddim_inv_slerp.json
fi

# ── Section 3: MoDiffAE + DDIM inversion + SLERP ─────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp
if _sec 3; then
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 1 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --blend_schedule $SCHEDULE --alpha_values $ALPHA --intersection_only --sampler ddim --ddim_inversion --transition_slerp
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
    cp $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100_slerp${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddim_inv_slerp.json
fi

# ── Section 4: MoDiffAE + DDIM inversion (no SLERP) ──────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100
if _sec 4; then
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 1 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --blend_schedule $SCHEDULE --alpha_values $ALPHA --intersection_only --sampler ddim --ddim_inversion
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100 --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
    cp $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddim-sample_ddim-inv-100${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddim_inv.json
fi

# ── Section 5: MoDiffAE + DDPM (5 repetitions) ───────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddpm-sample
if _sec 5; then
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 5 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --blend_schedule $SCHEDULE --alpha_values $ALPHA --intersection_only --sampler ddpm
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddpm-sample --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
    cp $MODEL_OUT_BASE/bench-$TYPE-$BENCH-intersection_static_alpha0.5_ddpm-sample${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddpm.json
fi
