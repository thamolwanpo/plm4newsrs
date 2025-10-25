#!/bin/bash

# Run with
# nohup ./eval.sh > /dev/null 2>&1 &

# --- Setup Logging ---
LOG_DIR="logs"
# Create the logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Define a single log file for this evaluation batch
GLOBAL_LOG="$LOG_DIR/evaluation_batch_log_$(date +%Y%m%d_%H%M%S).txt"

# A simple function to run and log a command
run_and_log() {
    COMMAND=$1
    LOG_FILE=$2
    
    # Print start time/command to terminal and log file
    echo "--- Starting: $COMMAND at $(date) ---" | tee -a "$LOG_FILE"
    
    # Execute the command, redirecting all output to the log file
    $COMMAND >> "$LOG_FILE" 2>&1
    EXIT_STATUS=$?
    
    # Print finish time/status to terminal and log file
    echo "--- Finished: $COMMAND with exit status $EXIT_STATUS at $(date) ---" | tee -a "$LOG_FILE"
    
    # STOP if a job fails (non-zero exit status)
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "FATAL ERROR: Previous evaluation job failed. Terminating script." | tee -a "$LOG_FILE"
        exit 1
    fi
}

# -------------------------------------------------------------
# --- Evaluation Commands ---
# -------------------------------------------------------------

# evaluate naml
echo "Starting NAML evaluations..." | tee -a "$GLOBAL_LOG"
# BERT finetune
run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_bert_finetune.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_bert_finetune_gossipcop.yaml" "$GLOBAL_LOG"

# BERT frozen
run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_bert_frozen.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_bert_frozen_gossipcop.yaml" "$GLOBAL_LOG"

# # Glove
# run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_glove.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/naml/naml_glove_gossipcop.yaml" "$GLOBAL_LOG"

# evaluate nrms
# echo "Starting NRMS evaluations..." | tee -a "$GLOBAL_LOG"
# BERT finetune 
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_finetune_gossipcop.yaml" "$GLOBAL_LOG"

# BERT frozen
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_frozen.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_bert_frozen_gossipcop.yaml" "$GLOBAL_LOG"

# # Glove
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml" "$GLOBAL_LOG"
# run_and_log "python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove_gossipcop.yaml" "$GLOBAL_LOG"

echo "All evaluation jobs in the batch are complete!" | tee -a "$GLOBAL_LOG"