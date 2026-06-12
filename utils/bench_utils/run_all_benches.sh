#!/usr/bin/env bash
# Run multiple benchmarks in sequence (temporal/static/transfer/alpha-ablation).
#
# Usage:
#   bash utils/bench_utils/run_all_benches.sh [MODEL_NAME] [CKPT] [SECTIONS] [eval_only]
#
#   bash utils/bench_utils/run_all_benches.sh                                   # all sections, default model
#   bash utils/bench_utils/run_all_benches.sh "" "" "2 3"                       # sections 2 and 3, default model
#   bash utils/bench_utils/run_all_benches.sh "" "" "1 3" eval_only             # eval-only, sections 1 and 3
#   bash utils/bench_utils/run_all_benches.sh mymodel 000699998                 # all sections, custom model
#   bash utils/bench_utils/run_all_benches.sh mymodel 000699998 "3 4" eval_only # sections 3 4, eval-only
#
# Add or remove lines in the <JOBS> section below to configure which benchmarks to run.
#
# NOTE: Always run from the project root so relative paths (./save, ./eval, etc.) resolve correctly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Global model/ckpt — override here or pass as args ────────────────────────
MODEL_NAME=${1:-modiffae_truebones_all}
CKPT=${2:-000599998}
SECTIONS=${3:-}
EVAL_ONLY=${4:-}
# ─────────────────────────────────────────────────────────────────────────────

run() {
    local script=$1
    local bench=$2
    local type=$3
    local sections=$4
    local model=$5
    local ckpt=$6
    local eval_only=$7
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Running: $script  BENCH=$bench  TYPE=$type  MODEL=$model  CKPT=$ckpt  SECTIONS=${sections:-all}  EVAL_ONLY=${eval_only:-no}"
    echo "════════════════════════════════════════════════════════"
    bash "$SCRIPT_DIR/$script" "$bench" "$type" "$sections" "$model" "$ckpt" "$eval_only"
    if [ $? -ne 0 ]; then
        echo "ERROR: $script $bench $type $sections $model $ckpt failed — aborting."
        exit 1
    fi
}

run_transfer() {
    local bench=$1
    local model=$2
    local ckpt=$3
    local eval_only=$4
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Running: transfer_bench.sh  BENCH=$bench  MODEL=$model  CKPT=$ckpt  EVAL_ONLY=${eval_only:-no}"
    echo "════════════════════════════════════════════════════════"
    bash "$SCRIPT_DIR/transfer_bench.sh" "$bench" "$model" "$ckpt" "$eval_only"
    if [ $? -ne 0 ]; then
        echo "ERROR: transfer_bench.sh $bench failed — aborting."
        exit 1
    fi
}

# run_alpha_ablation <BENCH> <TYPE> [SECTIONS] [MODEL] [CKPT] <SCHEDULE> [eval_only]
run_alpha_ablation() {
    local bench=$1
    local type=$2
    local sections=$3
    local model=$4
    local ckpt=$5
    local schedule=$6
    local eval_only=$7
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Running: alpha_ablation_bench.sh  BENCH=$bench  TYPE=$type  MODEL=$model  CKPT=$ckpt  SECTIONS=${sections:-all}  SCHEDULE=${schedule}  EVAL_ONLY=${eval_only:-no}"
    echo "════════════════════════════════════════════════════════"
    bash "$SCRIPT_DIR/alpha_ablation_bench.sh" "$bench" "$type" "$sections" "$model" "$ckpt" "$schedule" "$eval_only"
    if [ $? -ne 0 ]; then
        echo "ERROR: alpha_ablation_bench.sh $bench $type $schedule failed — aborting."
        exit 1
    fi
}

# ── JOBS ─────────────────────────────────────────────────────────────────────
# All helpers accept MODEL_NAME and CKPT so every job uses the model set above.
#
# Signatures:  eval_only is always last so it can be omitted when not needed
#   run                <script> <BENCH> <TYPE> [SECTIONS] [MODEL] [CKPT] [eval_only]
#   run_transfer       <BENCH> [MODEL] [CKPT] [eval_only]
#   run_alpha_ablation <BENCH> <TYPE> [SECTIONS] [MODEL] [CKPT] <SCHEDULE> [eval_only]
#                      Repeat once per schedule to ablate — each run is independent.
#
# Scripts: temporal_bench.sh | static_bench.sh | transfer_bench.sh | alpha_ablation_bench.sh

# Example: Temporal benchmark
# run temporal_bench.sh walk_run in_skel "$SECTIONS" $MODEL_NAME $CKPT
# Example: Static benchmark
# run static_bench.sh walk_run in_skel "$SECTIONS" $MODEL_NAME $CKPT
# Example: Transfer benchmark
# run_transfer all_walk $MODEL_NAME $CKPT

run_alpha_ablation walk_run in_skel "2 3 4" modiffae_truebones_all_globpool 000449998 linear
run_alpha_ablation walk_run in_skel "2 3 4" modiffae_truebones_all_globpool 000449998 ease_slope2

# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "All benchmarks completed."
