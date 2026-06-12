# There are 2 type of benchmarks: "in_skel" and "x_skel". The former requires the reference and target motions to share the same skeleton, while the latter does not have such a requirement. The "walk_walk" benchmark is an "in_skel" benchmark
# Moreover there are 2 other subtypes: "temporal" or "intersection".
# Follows an example for a "Temporal in_skel" benchmark, which is the "walk_walk" benchmark. The "intersection" benchmark is similar, except that the blend schedule is different (see the "blend_schedule" argument in the "mix.py" script)

# Usage: bash utils/bench_utils/temporal_bench.sh <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [SCHEDULE_ABLATION] [eval_only]
#   BENCH              : benchmark category, e.g. walk_run                     (required)
#   TYPE               : in_skel or x_skel                                     (default: in_skel)
#   SECTIONS           : quoted space-separated section numbers to run         (default: "" = all)
#                        e.g. "2 3" runs only sections 2 and 3
#   MODEL_NAME         : model directory name under ./save/                    (default: modiffae_truebones_all)
#   CKPT               : checkpoint iter, e.g. 000599998                       (default: 000599998)
#   SCHEDULE_ABLATION  : quoted space-separated schedule specs to ablate       (default: "ease")
#                        Each spec is one of:
#                          linear          → --blend_schedule linear
#                          ease            → --blend_schedule ease  (slope=1, default)
#                          ease_slope<N>   → --blend_schedule ease --ease_slope <N>
#                        e.g. "linear ease ease_slope2" runs all three variants
#   eval_only          : pass "eval_only" to skip generation                   (default: "")
#
# Sections:
#   1  NLA baseline
#   2  AnyTop + DDIM inversion + SLERP
#   3  MoDiffAE + DDIM inversion + SLERP
#   4  MoDiffAE + DDPM (5 repetitions)
BENCH=${1:?"Usage: $0 <BENCH> [TYPE] [SECTIONS] [MODEL_NAME] [CKPT] [SCHEDULE_ABLATION] [eval_only]  — e.g. $0 walk_run in_skel"}
TYPE=${2:-in_skel}
SECTIONS=${3:-}
# Current default model: modiffae_truebones_all / 000599998
MODEL_NAME=${4:-modiffae_truebones_all}
CKPT=${5:-000599998}
SCHEDULE_ABLATION=${6:-ease}
EVAL_ONLY=${7:-}

OVERLAP_LENGTH=30
# x_skel has stochastic tgt/transition zones (cross-skeleton DDIM inversion uses random noise),
# so multiple reps are needed. in_skel is fully deterministic with DDIM inversion → 1 rep suffices.
DDIM_REPS=$( [ "$TYPE" = "x_skel" ] && echo 5 || echo 1 )

REF_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/ref.txt
TGT_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/tgt.txt
GT_DIR=./dataset/truebones/zoo/truebones_processed/bvhs/

# ── Model versions ────────────────────────────────────────────────────────────
# Output base: ./save/samples_{MODEL_NAME}_{CKPT}_seed10/
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

# Returns 0 (true) if section N should run: either SECTIONS is empty (run all)
# or N appears in the SECTIONS list.
_sec() { [ -z "$SECTIONS" ] || echo " $SECTIONS " | grep -qw "$1"; }

# Parse a schedule spec into --blend_schedule / --ease_slope flags.
# Sets $_sched_flags (flags string) and $_sched_label (path-safe label).
#
#   linear        → --blend_schedule linear          label: linear
#   ease          → --blend_schedule ease             label: ease
#   ease_slope<N> → --blend_schedule ease --ease_slope <N>  label: ease_slope<N>
_parse_schedule() {
    local spec=$1
    case "$spec" in
        linear)
            _sched_flags="--blend_schedule linear"
            _sched_label="linear"
            ;;
        ease)
            _sched_flags="--blend_schedule ease"
            _sched_label="ease"
            ;;
        ease_slope*)
            local slope="${spec#ease_slope}"
            _sched_flags="--blend_schedule ease --ease_slope $slope"
            _sched_label="ease_slope${slope}"
            ;;
        *)
            echo "Unknown schedule spec: $spec" >&2
            exit 1
            ;;
    esac
}

# ── Section 1: NLA baseline ───────────────────────────────────────────────────
# Output: ./save/nla_blend/bench-$TYPE-$BENCH-temporal_<schedule>
if _sec 1; then
    for _spec in $SCHEDULE_ABLATION; do
        _parse_schedule "$_spec"
        _nla_out_dir="./save/nla_blend/bench-$TYPE-$BENCH-temporal_${_sched_label}"
        if [ "$EVAL_ONLY" != "eval_only" ]; then
            blender --background --python ./sample/nla_blend.py -- --overlap_length $OVERLAP_LENGTH $_sched_flags --ref_samples $REF_LIST --tgt_samples $TGT_LIST
        fi
        python3 -m eval.eval_truebones_blend --eval_gen_dir $_nla_out_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
        cp ${_nla_out_dir}${EVAL_SUFFIX}.json $BENCH_OUT/nla_${_sched_label}.json
    done
fi

# ── Section 2: AnyTop + DDIM inversion + SLERP ───────────────────────────────
# Output: $ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddim-sample_ddim-inv-100_slerp
if _sec 2; then
    for _spec in $SCHEDULE_ABLATION; do
        _parse_schedule "$_spec"
        _anytop_gen_dir="$ANYTOP_OUT_BASE/bench-$TYPE-$BENCH-temporal_${_sched_label}_ddim-sample_ddim-inv-100_slerp"
        if [ "$EVAL_ONLY" != "eval_only" ]; then
            python3 -m sample.mix --model_path $ANYTOP_PATH --num_repetitions 1 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $_sched_flags --sampler ddim --ddim_inversion --transition_slerp
        fi
        python3 -m eval.eval_truebones_blend --eval_gen_dir $_anytop_gen_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
        cp ${_anytop_gen_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${ANYTOP_NAME}_${ANYTOP_CKPT}_ddim_inv_slerp_${_sched_label}.json
    done
fi

# ── Section 3: MoDiffAE + DDIM inversion + SLERP ─────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddim-sample_ddim-inv-100_slerp
if _sec 3; then
    for _spec in $SCHEDULE_ABLATION; do
        _parse_schedule "$_spec"
        _model_gen_dir="$MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_${_sched_label}_ddim-sample_ddim-inv-100_slerp"
        if [ "$EVAL_ONLY" != "eval_only" ]; then
            python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions $DDIM_REPS --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $_sched_flags --sampler ddim --ddim_inversion --transition_slerp
        fi
        python3 -m eval.eval_truebones_blend --eval_gen_dir $_model_gen_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
        cp ${_model_gen_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddim_inv_slerp_${_sched_label}.json
    done
fi


# ── Section 4: MoDiffAE + DDPM (5 repetitions) ───────────────────────────────
# Output: $MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_<schedule>_ddpm-sample
if _sec 4; then
    for _spec in $SCHEDULE_ABLATION; do
        _parse_schedule "$_spec"
        _model_ddpm_dir="$MODEL_OUT_BASE/bench-$TYPE-$BENCH-temporal_${_sched_label}_ddpm-sample"
        if [ "$EVAL_ONLY" != "eval_only" ]; then
            python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 5 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode both --overlap_length $OVERLAP_LENGTH $_sched_flags --sampler ddpm
        fi
        python3 -m eval.eval_truebones_blend --eval_gen_dir $_model_ddpm_dir --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST
        cp ${_model_ddpm_dir}${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddpm_${_sched_label}.json
    done
fi
