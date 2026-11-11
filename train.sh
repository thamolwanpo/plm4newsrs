#!/bin/bash
# Train (clean + every poisoned set) and evaluate a batch of configs.
#
# Run with
#   nohup ./train.sh > /dev/null 2>&1 &
#
# DATA_ROOT overrides the fakeun-dataprep output folder of each config, e.g.
#   DATA_ROOT=../fakeun-dataprep/output/problem2/politifact ./train.sh
#
# MODE=references trains (and evaluates) only the clean references of every
# poisoned set (oracle + retrain per flag level, three seeds each), not the
# clean / poisoned models:
#   MODE=references nohup ./train.sh > /dev/null 2>&1 &

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
GLOBAL_LOG="$LOG_DIR/batch_log_$(date +%Y%m%d_%H%M%S).txt"

CONFIGS=(
    # configs/experiments/nrms/nrms_bert_frozen.yaml
    configs/experiments/naml/naml_bert_frozen_politifact.yaml
    configs/experiments/lstur/lstur_bert_frozen_politifact.yaml
    # configs/experiments/nrms/nrms_bert_frozen_politifact.yaml
    # configs/experiments/naml/naml_bert_frozen_gossipcop.yaml
    # configs/experiments/lstur/lstur_bert_frozen_gossipcop.yaml
    # configs/experiments/nrms/nrms_bert_frozen_gossipcop.yaml
    # configs/experiments/nrms/nrms_glove_gossipcop.yaml
    # configs/experiments/naml/naml_glove_gossipcop.yaml
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

if [ "$MODE" = "references" ]; then
    TRAIN_ARGS=(--train-references --skip-existing)
    EVAL_ARGS=(--models oracle retrain)
else
    TRAIN_ARGS=(--train-all)
    EVAL_ARGS=()
fi

for CONFIG in "${CONFIGS[@]}"; do
    run_and_log python scripts/train.py --config "$CONFIG" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
    run_and_log python scripts/evaluate.py --config "$CONFIG" --generate-report "${EVAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
done

echo "All jobs in the batch are complete!" | tee -a "$GLOBAL_LOG"
