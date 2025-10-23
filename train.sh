#!/bin/bash

# Run with
# nohup ./train.sh > /dev/null 2>&1 &

# --- 1. Define Log Directory and File ---
LOG_DIR="logs"
# Ensure the directory exists
mkdir -p "$LOG_DIR"

# Define a single log file path within the new directory
GLOBAL_LOG="$LOG_DIR/training_batch_log_$(date +%Y%m%d_%H%M%S).txt"

# A simple function to run and log a command
run_and_log() {
    COMMAND=$1
    LOG_FILE=$2
    echo "--- Starting: $COMMAND at $(date) ---" | tee -a "$LOG_FILE"
    # Execute the command, redirecting all output to the log file
    $COMMAND >> "$LOG_FILE" 2>&1
    EXIT_STATUS=$?
    echo "--- Finished: $COMMAND with exit status $EXIT_STATUS at $(date) ---" | tee -a "$LOG_FILE"
    
    # Optional: Stop if a job fails (non-zero exit status)
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "ERROR: Previous job failed. Exiting script." | tee -a "$LOG_FILE"
        exit 1
    fi
}

# --- 2. Rest of the script remains the same, using the new $GLOBAL_LOG path ---

# # train naml
# run_and_log "python scripts/train.py --config configs/experiments/naml/naml_bert_finetune_gossipcop.yaml --train-all --dataset gossipcop" "$GLOBAL_LOG"
# run_and_log "python scripts/train.py --config configs/experiments/naml/naml_bert_finetune.yaml --train-all --dataset politifact" "$GLOBAL_LOG"

# # BERT frozen
# run_and_log "python scripts/train.py --config configs/experiments/naml/naml_bert_frozen_gossipcop.yaml --train-all --dataset gossipcop" "$GLOBAL_LOG"
# run_and_log "python scripts/train.py --config configs/experiments/naml/naml_bert_frozen.yaml --train-all --dataset politifact" "$GLOBAL_LOG"

# # train nrms
# run_and_log "python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune_gossipcop.yaml --train-all --dataset gossipcop" "$GLOBAL_LOG"
# run_and_log "python scripts/train.py --config configs/experiments/nrms/nrms_bert_finetune.yaml --train-all --dataset politifact" "$GLOBAL_LOG"

# BERT frozen
run_and_log "python scripts/train.py --config configs/experiments/nrms/nrms_bert_frozen_gossipcop.yaml --train-all --dataset gossipcop" "$GLOBAL_LOG"
run_and_log "python scripts/train.py --config configs/experiments/nrms/nrms_bert_frozen.yaml --train-all --dataset politifact" "$GLOBAL_LOG"

echo "All training jobs in the batch are complete!" | tee -a "$GLOBAL_LOG"