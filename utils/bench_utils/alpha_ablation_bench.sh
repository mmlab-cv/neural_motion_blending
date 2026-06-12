# Temporal benchmark with a single configurable alpha-profile schedule.
# Identical to temporal_bench.sh except SCHEDULE is a fixed parameter instead of a loop.
#
# Usage: bash utils/bench_utils/alpha_ablation_bench.sh <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [SCHEDULE] [eval_only]
#   BENCH      : benchmark category, e.g. walk_run                     (required)
#   TYPE       : in_skel or x_skel                                     (default: in_skel)
#   SECTIONS   : quoted space-separated section numbers to run         (default: "" = all)
#                e.g. "2 3" runs only sections 2 and 3
#   MODEL_NAME : model directory name under ./save/                    (default: modiffae_truebones_all)
#   CKPT       : checkpoint iter, e.g. 000599998                       (default: 000599998)
#   SCHEDULE   : schedule spec for the alpha profile                   (default: ease)
#                One of:
#                  linear          → --blend_schedule linear
#                  ease            → --blend_schedule ease  (slope=1)
#                  ease_slope<N>   → --blend_schedule ease --ease_slope <N>
#   eval_only  : pass "eval_only" to skip generation                   (default: "")
#
# Sections:
#   1  NLA baseline
#   2  AnyTop + DDIM inversion + SLERP
#   3  MoDiffAE + DDIM inversion + SLERP
#   4  MoDiffAE + DDPM (5 repetitions)
BENCH=${1:?"Usage: $0 <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [SCHEDULE] [eval_only]  — e.g. $0 walk_run in_skel '' '' '' ease_slope2"}
TYPE=${2:-in_skel}
SECTIONS=${3:-}
MODEL_NAME=${4:-modiffae_truebones_all}
CKPT=${5:-000599998}
SCHEDULE=${6:-ease}
EVAL_ONLY=${7:-}

OVERLAP_LENGTH=30
DDIM_REPS=$( [ "$TYPE" = "x_skel" ] && echo 5 || echo 1 )

# Parse SCHEDULE into CLI flags and a path-safe label
case "$SCHEDULE" in
    linear)
        SCHED_FLAGS="--blend_schedule linear"
        SCHED_LABEL="linear"
        ;;
    ease)
        SCHED_FLAGS="--blend_schedule ease"
        SCHED_LABEL="ease"
        ;;
    ease_slope*)
        _slope="${SCHEDULE#ease_slope}"
        SCHED_FLAGS="--blend_schedule ease --ease_slope $_slope"
        SCHED_LABEL="ease_slope${_slope}"
        ;;
    *)
        echo "Unknown schedule: $SCHEDULE" >&2
        exit 1
        ;;
esac

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

BENCH_OUT=./bench/temporal-$TYPE-$BENCH
EVAL_SUFFIX="__source_bvh__feats_loc__rootrel"
mkdir -p $BENCH_OUT

_sec() { [ -z "$SECTIONS" ] || echo " $SECTIONS " | grep -qw "$1"; }

# ── Section 1: NLA baseline ───────────────────────────────────────────────────
# Output: ./save/nla_blend/bench-$TYPE-$BENCH-temporal_<schedule>
if _sec 1; then
    _out_dir="./save/nla_blend/bench-$TYPE-$BENCH-temporal_${SCHED_LABEL}"
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        blender --background --python ./sample/nla_blend.py -- --overlap_length $OVERLAP_LENGTH $SCHED_FLAGS --ref_samples $REF_LIST --tgt_samples $TGT_LIST
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $_out_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
    cp ${_out_dir}${EVAL_SUFFIX}.json $BENCH_OUT/nla_${SCHED_LABEL}.json
fi

# ── Section 2: AnyTop + DDIM inversion + SLERP ───────────────────────────────
# Output: $ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddim-sample_ddim-inv-100_slerp
if _sec 2; then
    _out_dir="$ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-temporal_${SCHED_LABEL}_ddim-sample_ddim-inv-100_slerp"
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $ANYTOP_PATH --num_repetitions 1 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $SCHED_FLAGS --sampler ddim --ddim_inversion --transition_slerp
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $_out_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
    cp ${_out_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${ANYTOP_NAME}_${ANYTOP_CKPT}_ddim_inv_slerp_${SCHED_LABEL}.json
fi

# ── Section 3: MoDiffAE + DDIM inversion + SLERP ─────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddim-sample_ddim-inv-100_slerp
if _sec 3; then
    _out_dir="$MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_${SCHED_LABEL}_ddim-sample_ddim-inv-100_slerp"
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions $DDIM_REPS --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $SCHED_FLAGS --sampler ddim --ddim_inversion --transition_slerp
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $_out_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
    cp ${_out_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddim_inv_slerp_${SCHED_LABEL}.json
fi

# ── Section 4: MoDiffAE + DDPM (5 repetitions) ───────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddpm-sample
if _sec 4; then
    _out_dir="$MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_${SCHED_LABEL}_ddpm-sample"
    if [ "$EVAL_ONLY" != "eval_only" ]; then
        python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 5 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $SCHED_FLAGS --sampler ddpm
    fi
    python3 -m eval.eval_truebones_blend --eval_gen_dir $_out_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
    cp ${_out_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddpm_${SCHED_LABEL}.json
fi
