#!/bin/bash
# Train (clean + every poisoned set) and evaluate a batch of configs.
#
# Run with
#   nohup ./train.sh > /dev/null 2>&1 &
#
# DRIVE_ROOT overrides data_root ($DRIVE_ROOT/data/{dataset}), base_dir
# ($DRIVE_ROOT/outputs/{dataset}), and LOG_DIR ($DRIVE_ROOT/logs), e.g.
#   DRIVE_ROOT=/content/drive/MyDrive/P2 ./train.sh
#
# DATA_ROOT overrides only the fakeun-dataprep output folder of each config, e.g.
#   DATA_ROOT=../fakeun-dataprep/output/problem2/politifact ./train.sh
#
# MODE=references trains (and evaluates) only the clean references of every
# poisoned set (oracle + retrain per flag level, three seeds each), not the
# clean / poisoned models; MODE=oracle only the oracle:
#   MODE=references nohup ./train.sh > /dev/null 2>&1 &
#   MODE=oracle POISON_SETS="A_r0.5_s1 A_r0.5_s2 A_r0.5_s3" nohup ./train.sh > /dev/null 2>&1 &
# With MODE=references / oracle, also:
#   POISON_SETS  poisoned sets to train references for (default: all in stage4/)
#   FLAG_LEVELS  flag levels of the retrain models, e.g. "flag_050" (default: all but 100)
#   NUM_SEEDS    seeds per file (default: 3)

POISON_SETS="${POISON_SETS:-A_r0.5_s1 A_r0.5_s2 A_r0.5_s3}"

if [ -n "${DRIVE_ROOT:-}" ]; then
    export DRIVE_ROOT
    LOG_DIR="${LOG_DIR:-$DRIVE_ROOT/logs}"
else
    LOG_DIR="${LOG_DIR:-logs}"
fi
mkdir -p "$LOG_DIR"
GLOBAL_LOG="$LOG_DIR/batch_log_$(date +%Y%m%d_%H%M%S).txt"

CONFIGS=(
    configs/experiments/naml/naml_bert_frozen_politifact.yaml
    configs/experiments/lstur/lstur_bert_frozen_politifact.yaml
    configs/experiments/nrms/nrms_bert_frozen_politifact.yaml
    # configs/experiments/lightgcn/lightgcn_bert_frozen_politifact.yaml
    # configs/experiments/naml/naml_bert_frozen_gossipcop.yaml
    # configs/experiments/lstur/lstur_bert_frozen_gossipcop.yaml
    # configs/experiments/nrms/nrms_bert_frozen_gossipcop.yaml
)

EXTRA_ARGS=()
if [ -n "$DATA_ROOT" ]; then
    EXTRA_ARGS=(--data-root "$DATA_ROOT")
fi

run_and_log() {
    echo "--- Starting: $* at $(date) ---" | tee -a "$GLOBAL_LOG"
    "$@" >> "$GLOBAL_LOG" 2>&1
    EXIT_STATUS=$?
    echo "--- Finished: $* with exit status $EXIT_STATUS at $(date) ---" | tee -a "$GLOBAL_LOG"
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "ERROR: Previous job failed. Exiting script." | tee -a "$GLOBAL_LOG"
        exit 1
    fi
}

if [ "$MODE" = "references" ] || [ "$MODE" = "oracle" ]; then
    TRAIN_ARGS=(--train-references --skip-existing --num-seeds "${NUM_SEEDS:-3}")
    if [ "$MODE" = "oracle" ]; then
        TRAIN_ARGS+=(--reference-types oracle)
        EVAL_ARGS=(--models oracle)
    else
        EVAL_ARGS=(--models oracle retrain)
    fi
    [ -n "$POISON_SETS" ] && TRAIN_ARGS+=(--poison-sets $POISON_SETS)
    [ -n "$FLAG_LEVELS" ] && TRAIN_ARGS+=(--flag-levels $FLAG_LEVELS)
else
    TRAIN_ARGS=(--train-all --skip-existing)
    EVAL_ARGS=()
fi

for CONFIG in "${CONFIGS[@]}"; do
    run_and_log python scripts/train.py --config "$CONFIG" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
    # run_and_log python scripts/evaluate.py --config "$CONFIG" --generate-report "${EVAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
done

echo "All jobs in the batch are complete!" | tee -a "$GLOBAL_LOG"