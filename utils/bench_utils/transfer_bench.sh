# Transfer benchmark: cross-skeleton motion transfer using control_mode=tgt.
# Always runs as x_skel. Only the overlapping frames [0, min(ref_len, tgt_len)) are evaluated.
#
# mix.py output dir suffix:  bench-x_skel-$BENCH-transfer_ddpm-sample
# eval: excludes self-pairs

# Usage: bash utils/bench_utils/transfer_bench.sh <BENCH> [MODEL_NAME] [CKPT] [eval_only]
#   BENCH      : benchmark category, e.g. walk_run                     (required)
#   MODEL_NAME : model directory name under ./save/                    (default: modiffae_truebones_all)
#   CKPT       : checkpoint iter, e.g. 000599998                       (default: 000599998)
#   eval_only  : pass "eval_only" to skip generation                   (default: "")
#
# Sections:
#   1  MoDiffAE + DDPM (5 repetitions, control_mode=tgt, x_skel)
BENCH=${1:?"Usage: $0 <BENCH> [MODEL_NAME] [CKPT] [eval_only]  — e.g. $0 walk_run"}
# Current default model: modiffae_truebones_all / 000599998
MODEL_NAME=${2:-modiffae_truebones_all}
CKPT=${3:-000599998}
EVAL_ONLY=${4:-}

TYPE=x_skel

REF_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/ref.txt
TGT_LIST=./eval/benchmarks/blend/$TYPE/$BENCH/tgt.txt
GT_DIR=./dataset/truebones/zoo/truebones_processed/bvhs/

# ── Model versions ────────────────────────────────────────────────────────────
MODEL_PATH=./save/$MODEL_NAME/model${CKPT}.pt
MODEL_OUT_BASE=./save/$MODEL_NAME/samples_${MODEL_NAME}_${CKPT}_seed10
# ─────────────────────────────────────────────────────────────────────────────

BENCH_OUT=./bench/transfer-$TYPE-$BENCH
EVAL_SUFFIX="__source_bvh__feats_loc__rootrel"
mkdir -p $BENCH_OUT

# ── Section 1: MoDiffAE + DDPM (5 repetitions, control_mode=tgt) ─────────────
# Output: $MODEL_OUT_BASE/bench-x_skel-$BENCH-tgt_ddpm-sample
if [ "$EVAL_ONLY" != "eval_only" ]; then
    python3 -m sample.mix --model_path $MODEL_PATH --num_repetitions 5 --ref_samples $REF_LIST --tgt_samples $TGT_LIST --control_mode tgt --sampler ddpm
fi
python3 -m eval.eval_truebones_blend --eval_gen_dir $MODEL_OUT_BASE/bench-$TYPE-$BENCH-transfer_ddpm-sample --eval_gt_dir $GT_DIR --source bvh --feats loc --root_relative --ref_samples $REF_LIST --tgt_samples $TGT_LIST --exclude_self_pairs
cp $MODEL_OUT_BASE/bench-$TYPE-$BENCH-transfer_ddpm-sample${EVAL_SUFFIX}.json $BENCH_OUT/${MODEL_NAME}_${CKPT}_ddpm.json
