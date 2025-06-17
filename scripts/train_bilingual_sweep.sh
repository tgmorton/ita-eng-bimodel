#!/bin/bash
#SBATCH --job-name=bilingual-model-sweep      # A descriptive job name
#SBATCH --partition=general_gpu_p6000       # The partition to run on
#SBATCH --array=0-5                         # Create a job array for the 6 configurations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:00:00                   # Time limit (D-HH:MM:SS)
#SBATCH --output=../logs/%x-%A_%a.out      # Unique standard output log for each array task
#SBATCH --error=../logs/%x-%A_%a.err       # Unique standard error log for each array task

# Exit on any error
set -e

# --- Environment Setup ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8

# --- Define Paths ---
HOST_PROJECT_DIR="/home/AD/thmorton/ita-eng-bimodel"
HOST_SIF_PATH="/home/AD/thmorton/ita-eng-bimodel/italian_llm_env.sif"

# --- Define the array of experiment config files ---
# This array holds the path to each experiment we want to run.
CONFIG_FILES=(
    "configs/10_25_it_eng.yaml"
    "configs/25_25_it_eng.yaml"
    "configs/50_25_it_eng.yaml"
    "configs/10_25_eng_it.yaml"
    "configs/25_25_eng_it.yaml"
    "configs/50_25_eng_it.yaml"
)

# --- Select the config file for the current array task ---
CURRENT_CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$CURRENT_CONFIG" ]; then
    echo "ERROR: No config file found for array task ID $SLURM_ARRAY_TASK_ID."
    exit 1
fi

# --- Define dynamic output directory based on the config name ---
CONFIG_NAME=$(basename -s .yaml "$CURRENT_CONFIG")
OUTPUT_DIR="output/bilingual_sweep/${CONFIG_NAME}"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Config File for this task: ${CURRENT_CONFIG}"
echo "Output Directory for this task: ${OUTPUT_DIR}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"
mkdir -p "${HOST_PROJECT_DIR}/${OUTPUT_DIR}"


# --- Training Script Execution ---
echo "Starting Python training script inside Singularity container..."

# Set PyTorch CUDA Allocator Config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container.
# The script will use the settings from the YAML file, but we override
# the output directory to keep runs separate.
srun singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_SIF_PATH}" \
    bash -c "cd /workspace && python -m src.train \
        --config-file ${CURRENT_CONFIG} \
        --output_dir ${OUTPUT_DIR}"

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"
