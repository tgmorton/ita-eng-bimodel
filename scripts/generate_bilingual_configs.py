# scripts/generate_bilingual_configs.py

import yaml
from pathlib import Path
import math
import numpy as np
from datasets import load_from_disk
from rich.console import Console

# --- Re-used Core Logic from generate_schedule.py ---

def calculate_steps(dataset_path: Path, chunk_size: int, effective_batch_size: int, num_epochs: int) -> int:
    """Calculates the total number of training steps for a given dataset."""
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
    base_dir = Path("~/ita-eng-bimodel").expanduser()
    tokenized_data_dir = base_dir / "data/tokenized"
    tokenizer_base_dir = base_dir / "tokenizer"
    configs_output_dir = base_dir / "configs"
    configs_output_dir.mkdir(exist_ok=True)

    # These parameters are shared across all experiment configs
    batch_size = 8
    gradient_accumulation_steps = 16
    num_epochs = 1
    chunk_size = 1024
    effective_batch_size = batch_size * gradient_accumulation_steps

    bilingual_setups = [
        {'name': '10_25_it_eng', 'l1_lang': 'italian', 'l1_size': '10M', 'l2_lang': 'english', 'l2_size': '25M', 'l1_checkpoints': 20, 'l2_checkpoints': 50},
        {'name': '25_25_it_eng', 'l1_lang': 'italian', 'l1_size': '25M', 'l2_lang': 'english', 'l2_size': '25M', 'l1_checkpoints': 50, 'l2_checkpoints': 50},
        {'name': '50_25_it_eng', 'l1_lang': 'italian', 'l1_size': '50M', 'l2_lang': 'english', 'l2_size': '25M', 'l1_checkpoints': 100, 'l2_checkpoints': 50},
        {'name': '10_25_eng_it', 'l1_lang': 'english', 'l1_size': '10M', 'l2_lang': 'italian', 'l2_size': '25M', 'l1_checkpoints': 20, 'l2_checkpoints': 50},
        {'name': '25_25_eng_it', 'l1_lang': 'english', 'l1_size': '25M', 'l2_lang': 'italian', 'l2_size': '25M', 'l1_checkpoints': 50, 'l2_checkpoints': 50},
        {'name': '50_25_eng_it', 'l1_lang': 'english', 'l1_size': '50M', 'l2_lang': 'italian', 'l2_size': '25M', 'l1_checkpoints': 100, 'l2_checkpoints': 50},
    ]

    for setup in bilingual_setups:
        name = setup['name']
        console.print(f"\n[bold green]===== Generating config for: {name} =====[/bold green]")

        # --- Calculate Schedule ---
        l1_path = tokenized_data_dir / name / "l1_train"
        l2_path = tokenized_data_dir / name / "l2_train"

        l1_total_steps = calculate_steps(l1_path, chunk_size, effective_batch_size, num_epochs)
        l2_total_steps = calculate_steps(l2_path, chunk_size, effective_batch_size, num_epochs)

        l1_schedule = generate_phase_schedule(l1_total_steps, setup['l1_checkpoints'])
        l2_schedule_raw = generate_phase_schedule(l2_total_steps, setup['l2_checkpoints'])
        l2_schedule_offset = {s + l1_total_steps for s in l2_schedule_raw}
        full_schedule = sorted([int(s) for s in (l1_schedule | {l1_total_steps} | l2_schedule_offset)])

        # --- Build YAML Config Dictionary ---
        config_dict = {
            # Essential Paths
            "l1_dataset_path": str(l1_path),
            "l2_dataset_path": str(l2_path),
            "output_dir": f"output/{name}",
            "tokenizer_path": str(tokenizer_base_dir / name),
            "architectures_path": "configs/model_architectures.yaml",

            # Model Configuration
            "train_from_scratch": True,
            "model_arch_type": "gpt2",
            # IMPORTANT: You must change this to a valid key from your model_architectures.yaml
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
            # save_steps is a fallback; the schedule is primary
            "save_steps": 500,
            "checkpoint_schedule": full_schedule,
        }

        # --- Write to File ---
        output_filepath = configs_output_dir / f"{name}.yaml"
        with open(output_filepath, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False, indent=2)

        console.print(f"Successfully wrote config to [yellow]{output_filepath}[/yellow]")

if __name__ == "__main__":
    generate_configs()
