# scripts/generate_bilingual_configs.py

import yaml
from pathlib import Path
import math
import numpy as np
from datasets import load_from_disk
from rich.console import Console

# --- Re-used Core Logic ---

def calculate_steps(dataset_path: Path, chunk_size: int, effective_batch_size: int, num_epochs: int) -> int:
    """Calculates the total number of training steps for a given dataset."""
    # Correctly expand user for local execution of this script
    dataset_path = dataset_path.expanduser()
    if not dataset_path.exists():
        print(f"Warning: Dataset path not found, cannot calculate steps: {dataset_path}")
        return 0
    dataset = load_from_disk(str(dataset_path))
    num_chunks = sum(math.ceil(len(ids) / chunk_size) for ids in dataset['input_ids'])
    steps_per_epoch = math.ceil(num_chunks / effective_batch_size)
    return steps_per_epoch * num_epochs

def generate_phase_schedule(total_steps: int, target_checkpoints: int) -> set:
    """Generates a checkpoint schedule for a single training phase."""
    if total_steps <= 0 or target_checkpoints <= 0:
        return set()
    log_steps = set()
    step = 1
    while step < total_steps / 2 and step < 1024:
        log_steps.add(step)
        step *= 2
    even_steps = set(np.linspace(1, total_steps, num=target_checkpoints, dtype=int))
    return log_steps | even_steps

# --- Main Configuration Generation Logic ---

def generate_configs():
    """
    Generates all the necessary YAML configuration files for the bilingual experiments.
    """
    console = Console()
    console.print("[bold cyan]--- Generating Bilingual Experiment Configurations ---[/bold cyan]")

    # --- Static Parameters ---
    # Define paths RELATIVE to the project root for use inside the container
    tokenized_data_dir = Path("data/tokenized")
    tokenizer_base_dir = Path("tokenizer")
    configs_output_dir = Path("configs")
    configs_output_dir.mkdir(exist_ok=True)

    # These parameters are shared across all experiment configs
    batch_size = 8
    gradient_accumulation_steps = 16
    num_epochs = 1
    chunk_size = 1024
    effective_batch_size = batch_size * gradient_accumulation_steps

    bilingual_setups = [
        {'name': '10_25_it_eng', 'l1_checkpoints': 20, 'l2_checkpoints': 50},
        {'name': '25_25_it_eng', 'l1_checkpoints': 50, 'l2_checkpoints': 50},
        {'name': '50_25_it_eng', 'l1_checkpoints': 100, 'l2_checkpoints': 50},
        {'name': '10_25_eng_it', 'l1_checkpoints': 20, 'l2_checkpoints': 50},
        {'name': '25_25_eng_it', 'l1_checkpoints': 50, 'l2_checkpoints': 50},
        {'name': '50_25_eng_it', 'l1_checkpoints': 100, 'l2_checkpoints': 50},
    ]

    for setup in bilingual_setups:
        name = setup['name']
        console.print(f"\n[bold green]===== Generating config for: {name} =====[/bold green]")

        # --- Calculate Schedule (using absolute paths for local calculation) ---
        # Note: The calculation needs the absolute path to load the dataset locally
        l1_path_local = Path("~/ita-eng-bimodel").expanduser() / tokenized_data_dir / name / "l1_train"
        l2_path_local = Path("~/ita-eng-bimodel").expanduser() / tokenized_data_dir / name / "l2_train"

        l1_total_steps = calculate_steps(l1_path_local, chunk_size, effective_batch_size, num_epochs)
        l2_total_steps = calculate_steps(l2_path_local, chunk_size, effective_batch_size, num_epochs)

        l1_schedule = generate_phase_schedule(l1_total_steps, setup['l1_checkpoints'])
        l2_schedule_raw = generate_phase_schedule(l2_total_steps, setup['l2_checkpoints'])
        l2_schedule_offset = {s + l1_total_steps for s in l2_schedule_raw}
        full_schedule = sorted([int(s) for s in (l1_schedule | {l1_total_steps} | l2_schedule_offset)])

        # --- Build YAML Config Dictionary (using RELATIVE paths) ---
        config_dict = {
            # Essential Paths (now relative)
            "l1_dataset_path": str(tokenized_data_dir / name / "l1_train"),
            "l2_dataset_path": str(tokenized_data_dir / name / "l2_train"),
            "output_dir": f"output/{name}",
            "tokenizer_path": str(tokenizer_base_dir / name),
            "architectures_path": "configs/model_architectures.yaml",

            # Model Configuration
            "train_from_scratch": True,
            "model_arch_type": "gpt2",
            "model_size_tag": "gpt2-100m",

            # Training Hyperparameters
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "num_warmup_steps": 100,

            # Hardware & Precision
            "use_amp": True,
            "num_workers": 4,

            # Control & Reproducibility
            "seed": 42,

            # Logging & Saving
            "logging_steps": 100,
            "save_steps": 500,
            "checkpoint_schedule": full_schedule,
        }

        # --- Write to File ---
        # Use the local path for writing the config file
        output_filepath = Path.home() / "ita-eng-bimodel" / "configs" / f"{name}.yaml"
        with open(output_filepath, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False, indent=2)

        console.print(f"Successfully wrote config to [yellow]{output_filepath}[/yellow]")

if __name__ == "__main__":
    generate_configs()