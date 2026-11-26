#!/usr/bin/env bash

set -euo pipefail

CONFIG="${CONFIG:-configs/experiments/lightgcn/lightgcn_bert_frozen_politifact.yaml}"
POISON_SETS="${POISON_SETS:-A_r0.5_s1 A_r0.5_s2 A_r0.5_s3}"
NUM_SEEDS="${NUM_SEEDS:-3}"
if [ -n "${DRIVE_ROOT:-}" ]; then
    export DRIVE_ROOT
    LOG_DIR="${LOG_DIR:-$DRIVE_ROOT/logs}"
else
    LOG_DIR="${LOG_DIR:-logs}"
fi

read -r -a POISON_SET_LIST <<< "$POISON_SETS"
if [ "${#POISON_SET_LIST[@]}" -eq 0 ]; then
    echo "ERROR: POISON_SETS must contain at least one set." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
GLOBAL_LOG="$LOG_DIR/oracle_and_poisoned_$(date +%Y%m%d_%H%M%S).txt"

EXTRA_ARGS=()
if [ -n "${DATA_ROOT:-}" ]; then
    EXTRA_ARGS=(--data-root "$DATA_ROOT")
fi

run_and_log() {
    echo "--- Starting: $* at $(date) ---" | tee -a "$GLOBAL_LOG"
    if "$@" >> "$GLOBAL_LOG" 2>&1; then
        echo "--- Finished: $* successfully at $(date) ---" | tee -a "$GLOBAL_LOG"
    else
        EXIT_STATUS=$?
        echo "--- Failed: $* with exit status $EXIT_STATUS at $(date) ---" | tee -a "$GLOBAL_LOG"
        exit "$EXIT_STATUS"
    fi
}

run_and_log python scripts/train.py \
    --config "$CONFIG" \
    --train-references \
    --reference-types oracle \
    --poison-sets "${POISON_SET_LIST[@]}" \
    --num-seeds "$NUM_SEEDS" \
    --skip-existing \
    "${EXTRA_ARGS[@]}"

MODEL_NAMES=()
for POISON_SET in "${POISON_SET_LIST[@]}"; do
    run_and_log python scripts/train.py \
        --config "$CONFIG" \
        --model-type poisoned \
        --poison-set "$POISON_SET" \
        --skip-existing \
        "${EXTRA_ARGS[@]}"
    MODEL_NAMES+=("oracle_${POISON_SET}" "poisoned_${POISON_SET}")
done

run_and_log python scripts/evaluate.py \
    --config "$CONFIG" \
    --generate-report \
    --models "${MODEL_NAMES[@]}" \
    "${EXTRA_ARGS[@]}"

echo "Oracle and poisoned model jobs are complete. Log: $GLOBAL_LOG" | tee -a "$GLOBAL_LOG"
