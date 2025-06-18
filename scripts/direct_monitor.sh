#!/bin/bash

# --- Direct Execution Script for Local GPU Evaluation ---
# This script runs the full evaluation monitor for a given model and tokenizer,
# now pointing to a single directory for all priming data.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_dir_name> <tokenizer_name> [eval_cases...]"
    echo "Example (all cases): ./run_local_eval.sh 50_25_it_eng 50_25_it_eng"
    echo "Example (priming only): ./run_local_eval.sh 50_25_it_eng 50_25_it_eng priming"
    exit 1
fi

# --- Argument Parsing ---
MODEL_DIR_NAME=$1   # e.g., "50_25_it_eng"
TOKENIZER_NAME=$2 # e.g., "50_25_it_eng"
shift 2
EVAL_CASES=("$@") # All remaining arguments are treated as eval cases

# === Environment Setup ===
echo "=== Script Started: $(date) ==="
echo "Target Model Dir: ${MODEL_DIR_NAME}"
echo "Using Tokenizer:  ${TOKENIZER_NAME}"

module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Host and Container Paths ---
# !!! UPDATE THIS TO YOUR PROJECT'S ROOT DIRECTORY !!!
HOST_PROJECT_DIR="/home/AD/thmorton/ita-eng-bimodel"
HOST_SIF_PATH="${HOST_PROJECT_DIR}/italian_llm_env.sif"

# Data directories on the host
HOST_DATA_DIR="${HOST_PROJECT_DIR}/data"
HOST_MODELS_DIR="${HOST_PROJECT_DIR}/output"
HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/results"
HOST_TOKENIZER_DIR="${HOST_PROJECT_DIR}/tokenizer"

# --- UPDATED: Simplified paths for evaluation data ---
HOST_SURPRISAL_DIR="${HOST_PROJECT_DIR}/evaluation/data/italian_null_subject"
HOST_PRIMING_DIR="${HOST_PROJECT_DIR}/evaluation/data/priming" # Consolidated priming data

# Paths inside the container
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_MODELS_DIR="/models"
CONTAINER_RESULTS_DIR="/results"
CONTAINER_TOKENIZER_DIR="/workspace/tokenizer"
CONTAINER_SURPRISAL_DIR="/surprisal_data"
CONTAINER_PRIMING_DIR="/priming_data" # Single container path for priming

# --- Process Optional Arguments ---
EVAL_ARGS=""
if [ ${#EVAL_CASES[@]} -gt 0 ]; then
    EVAL_ARGS="--eval_cases ${EVAL_CASES[*]}"
fi

# --- Preparations ---
echo "Project Directory: ${HOST_PROJECT_DIR}"
if [ ! -d "${HOST_MODELS_DIR}/${MODEL_DIR_NAME}" ]; then echo "ERROR: Model directory not found: ${HOST_MODELS_DIR}/${MODEL_DIR_NAME}"; exit 1; fi
if [ ! -d "${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}" ]; then echo "ERROR: Tokenizer directory not found: ${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}"; exit 1; fi
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "${HOST_PRIMING_DIR}" ]; then echo "ERROR: Priming data directory not found at ${HOST_PRIMING_DIR}"; exit 1; fi
mkdir -p "${HOST_RESULTS_DIR}"

# === Monitor Script Execution ===
echo "Starting Python monitor.py inside Singularity..."
echo "Running cases: ${EVAL_CASES[*]:-"all"}"

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_MODELS_DIR}":"${CONTAINER_MODELS_DIR}" \
    -B "${HOST_RESULTS_DIR}":"${CONTAINER_RESULTS_DIR}" \
    -B "${HOST_SURPRISAL_DIR}":"${CONTAINER_SURPRISAL_DIR}" \
    -B "${HOST_PRIMING_DIR}":"${CONTAINER_PRIMING_DIR}" \
    "${HOST_SIF_PATH}" \
    bash -c "cd ${CONTAINER_WORKSPACE} && python3 -m evaluation.monitor \
        --model_parent_dir \"${CONTAINER_MODELS_DIR}/${MODEL_DIR_NAME}\" \
        --output_base_dir \"${CONTAINER_RESULTS_DIR}\" \
        --tokenizer_path \"${CONTAINER_WORKSPACE}/tokenizer/${TOKENIZER_NAME}\" \
        --surprisal_data_dir \"${CONTAINER_SURPRISAL_DIR}\" \
        --priming_data_path \"${CONTAINER_PRIMING_DIR}\" \
        --perplexity_data_base_path \"${CONTAINER_DATA_DIR}/tokenized\" \
        ${EVAL_ARGS}"

# === Script Completion ===
echo "=== Script Finished: $(date) ==="