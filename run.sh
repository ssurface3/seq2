#!/bin/bash

# ==============================================================================
# DeBERTa NER Pipeline Launcher
# Automates: Environment Check -> Training/Optuna Search -> Inference
# ==============================================================================

TRAIN_SCRIPT="train_advanced.py"
PREDICT_SCRIPT="predict_.py"
TRAIN_DATA="train.tsv"
TEST_DATA="test_no_answers.tsv"
OUTPUT_MODEL_DIR="deberta-optimized-model"
SUBMISSION_FILE="submission_deberta.tsv"


log() {
    echo -e "\033[1;34m[NER-PIPE]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
    exit 1
}

log "Checking environment..."


if ! command -v python3 &> /dev/null; then
    error "Python3 could not be found."
fi


if [ ! -f "$TRAIN_DATA" ]; then
    error "Training data '$TRAIN_DATA' not found. Please place it in this directory."
fi

if [ ! -f "$TEST_DATA" ]; then
    error "Test data '$TEST_DATA' not found. Please place it in this directory."
fi


if command -v nvidia-smi &> /dev/null; then
    log "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log "WARNING: No NVIDIA GPU detected. Training will be very slow on CPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log "Starting Training Process (this includes Optuna HPO)..."
log "Executing $TRAIN_SCRIPT..."


python3 "$TRAIN_SCRIPT"

if [ $? -eq 0 ]; then
    log "Training completed successfully."
else
    error "Training failed. Check the error logs above."
fi


if [ ! -d "$OUTPUT_MODEL_DIR" ]; then
    error "Model directory '$OUTPUT_MODEL_DIR' was not created. Training might have failed silently."
fi

log "Starting Inference Process..."
log "Executing $PREDICT_SCRIPT..."

python3 "$PREDICT_SCRIPT"

if [ $? -eq 0 ]; then
    log "Inference completed."
else
    error "Inference failed."
fi

if [ -f "$SUBMISSION_FILE" ]; then
    log "SUCCESS! Pipeline finished."
    log "Submission file generated at: $(pwd)/$SUBMISSION_FILE"
    echo -e "\n--- Preview of Submission ---"
    head -n 5 "$SUBMISSION_FILE"
    echo "..."
else
    error "Pipeline finished, but '$SUBMISSION_FILE' was not found."
fi