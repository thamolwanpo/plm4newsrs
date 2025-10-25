#!/bin/bash

# Run with
# nohup ./unlearn.sh > /dev/null 2>&1 &

# --- Setup Logging ---
LOG_DIR="logs"
# Create the logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Define a single log file for this unlearning batch
GLOBAL_LOG="$LOG_DIR/unlearning_batch_log_$(date +%Y%m%d_%H%M%S).txt"

# A simple function to run and log a command
run_and_log() {
    COMMAND=$1
    LOG_FILE=$2
    
    echo "--- Starting: $COMMAND at $(date) ---" | tee -a "$LOG_FILE"
    
    # Execute the command, redirecting all output to the log file
    # We use 'eval' here because the command string contains backslashes for line continuation
    eval "$COMMAND" >> "$LOG_FILE" 2>&1
    EXIT_STATUS=$?
    
    echo "--- Finished: $COMMAND with exit status $EXIT_STATUS at $(date) ---" | tee -a "$LOG_FILE"
    
    # STOP if a job fails (non-zero exit status)
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "FATAL ERROR: Previous unlearning job failed (Exit Status $EXIT_STATUS). Terminating script." | tee -a "$LOG_FILE"
        exit 1
    fi
}

# -------------------------------------------------------------
# --- Common Parameters ---
# -------------------------------------------------------------

BASE_CMD="python scripts/run_unlearning_experiments.py"
COMMON_ARGS="--data-path data/politifact/train_poisoned.csv \
    --ratios 0.01 0.05 0.10 0.20 0.30 \
    --num-trials 3 \
    --save-summary"

# Config paths for each method
FIRST_ORDER_CONFIG="configs/experiments/unlearning/first_order.yaml"
GRADIENT_ASCENT_CONFIG="configs/experiments/unlearning/gradient_ascent.yaml"

# -------------------------------------------------------------
# --- Define Experiments ---
# -------------------------------------------------------------

declare -a EXPERIMENTS=(
    # naml bert finetune, first_order
    # "--model-configs configs/experiments/naml/naml_bert_finetune.yaml \
    #  --checkpoints outputs/politifact/naml_model/bert_finetune/checkpoints/poisoned-epoch=03-val_auc=0.7329.ckpt \
    #  --methods first_order \
    #  --unlearn-configs $FIRST_ORDER_CONFIG"

    # naml bert finetune, gradient_ascent
    # "--model-configs configs/experiments/naml/naml_bert_finetune.yaml \
    #  --checkpoints outputs/politifact/naml_model/bert_finetune/checkpoints/poisoned-epoch=03-val_auc=0.7329.ckpt \
    #  --methods gradient_ascent \
    #  --unlearn-configs $GRADIENT_ASCENT_CONFIG"

    # naml bert frozen, first_order
    "--model-configs configs/experiments/naml/naml_bert_frozen.yaml \
     --checkpoints outputs/politifact/naml_model/bert_frozen/checkpoints/poisoned-epoch=03-val_auc=0.7625.ckpt \
     --methods first_order \
     --unlearn-configs $FIRST_ORDER_CONFIG"

    # naml bert frozen, gradient_ascent
    "--model-configs configs/experiments/naml/naml_bert_frozen.yaml \
     --checkpoints outputs/politifact/naml_model/bert_frozen/checkpoints/poisoned-epoch=03-val_auc=0.7625.ckpt \
     --methods gradient_ascent \
     --unlearn-configs $GRADIENT_ASCENT_CONFIG"
)

# -------------------------------------------------------------
# --- Run Experiments ---
# -------------------------------------------------------------

echo "Starting Unlearning Experiments Batch ($(date))." | tee -a "$GLOBAL_LOG"
echo "Total experiments to run: ${#EXPERIMENTS[@]}" | tee -a "$GLOBAL_LOG"

for EXPERIMENT_ARGS in "${EXPERIMENTS[@]}"; do
    # Assemble the full command string
    FULL_COMMAND="$BASE_CMD $EXPERIMENT_ARGS $COMMON_ARGS"
    
    # Run the command through the logging function
    run_and_log "$FULL_COMMAND" "$GLOBAL_LOG"
done

echo "All unlearning experiments in the batch are complete!" | tee -a "$GLOBAL_LOG"