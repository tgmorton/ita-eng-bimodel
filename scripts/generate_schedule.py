# scripts/generate_schedule.py

import math
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import typer
import yaml
from datasets import load_from_disk
from pydantic import BaseModel, field_validator
from rich.console import Console
from rich.table import Table


# --- Configuration Models ---

class BilingualConfig(BaseModel):
    name: str
    l1_lang: str
    l1_size: str
    l2_lang: str
    l2_size: str
    l1_checkpoints: int
    l2_checkpoints: int


# --- Core Generation Functions ---

def calculate_steps(dataset_path: Path, chunk_size: int, effective_batch_size: int, num_epochs: int,
                    console: Console) -> int:
    """Calculates the total number of training steps for a given dataset with debugging."""
    console.print(f"\n[bold blue]-- Debugging path: {dataset_path} --[/bold blue]")

    dataset_path = dataset_path.expanduser()

    if not dataset_path.exists():
        console.print(f"[bold red]  - FAIL: Path does not exist.[/bold red]")
        return 0

    console.print(f"[green]  - SUCCESS: Path exists.[/green]")

    try:
        dataset = load_from_disk(str(dataset_path))
        console.print(f"[green]  - SUCCESS: Loaded dataset with {len(dataset):,} samples.[/green]")

        if 'input_ids' not in dataset.column_names:
            console.print(f"[bold red]  - FAIL: 'input_ids' column not found in dataset.[/bold red]")
            return 0

        num_chunks = sum(math.ceil(len(ids) / chunk_size) for ids in dataset['input_ids'])
        console.print(f"  - Calculated [yellow]{num_chunks:,}[/yellow] total chunks.")

        steps_per_epoch = math.ceil(num_chunks / effective_batch_size)
        console.print(f"  - Calculated [yellow]{steps_per_epoch:,}[/yellow] steps per epoch.")

        total_steps = steps_per_epoch * num_epochs
        console.print(f"  - Total steps for {num_epochs} epochs: [bold yellow]{total_steps:,}[/bold yellow]")

        return total_steps

    except Exception as e:
        console.print(f"[bold red]  - FAIL: An error occurred: {e}[/bold red]")
        return 0


def generate_phase_schedule(total_steps: int, target_checkpoints: int) -> Set[int]:
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


# --- Typer App for CLI ---

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
        base_data_dir: Path = typer.Option("~/ita-eng-bimodel/data/tokenized",
                                           help="Base directory for tokenized datasets."),
        batch_size: int = typer.Option(8, help="Batch size per device."),
        gradient_accumulation_steps: int = typer.Option(16, help="Gradient accumulation steps."),
        num_epochs: int = typer.Option(3, help="Total training epochs."),
        chunk_size: int = typer.Option(1024, help="The chunk size used for tokenizing."),
):
    """
    Generates tailored checkpoint schedules for bilingual training configurations.
    """
    effective_batch_size = batch_size * gradient_accumulation_steps

    bilingual_configs: List[BilingualConfig] = [
        BilingualConfig(name="10_25_it_eng", l1_lang="italian", l1_size="10M", l2_lang="english", l2_size="25M",
                        l1_checkpoints=10, l2_checkpoints=20),
        BilingualConfig(name="25_25_it_eng", l1_lang="italian", l1_size="25M", l2_lang="english", l2_size="25M",
                        l1_checkpoints=20, l2_checkpoints=20),
        BilingualConfig(name="50_25_it_eng", l1_lang="italian", l1_size="50M", l2_lang="english", l2_size="25M",
                        l1_checkpoints=30, l2_checkpoints=20),
        BilingualConfig(name="10_25_eng_it", l1_lang="english", l1_size="10M", l2_lang="italian", l2_size="25M",
                        l1_checkpoints=10, l2_checkpoints=20),
        BilingualConfig(name="25_25_eng_it", l1_lang="english", l1_size="25M", l2_lang="italian", l2_size="25M",
                        l1_checkpoints=20, l2_checkpoints=20),
        BilingualConfig(name="50_25_eng_it", l1_lang="english", l1_size="50M", l2_lang="italian", l2_size="25M",
                        l1_checkpoints=30, l2_checkpoints=20),
    ]

    console.print("[bold cyan]--- Generating Bilingual Checkpoint Schedules ---[/bold cyan]")

    for config in bilingual_configs:
        console.print(f"\n[bold green]===== Processing Configuration: {config.name} =====[/bold green]")

        l1_path = base_data_dir / config.name / "l1_train"
        l2_path = base_data_dir / config.name / "l2_train"

        l1_total_steps = calculate_steps(l1_path, chunk_size, effective_batch_size, num_epochs, console)
        l2_total_steps = calculate_steps(l2_path, chunk_size, effective_batch_size, num_epochs, console)

        table = Table(title=f"Results for {config.name}")
        table.add_column("Phase", style="dim")
        table.add_column("Details", style="magenta")
        table.add_column("Total Steps", justify="right", style="yellow")
        table.add_row("L1", f"{config.l1_lang.capitalize()} {config.l1_size}", f"{l1_total_steps:,}")
        table.add_row("L2", f"{config.l2_lang.capitalize()} {config.l2_size}", f"{l2_total_steps:,}")
        console.print(table)

        l1_schedule = generate_phase_schedule(l1_total_steps, config.l1_checkpoints)
        l2_schedule_raw = generate_phase_schedule(l2_total_steps, config.l2_checkpoints)

        l2_schedule_offset = {s + l1_total_steps for s in l2_schedule_raw}

        # --- CORRECTED TYPE CASTING ---
        # Convert all numbers to standard Python integers before dumping to YAML
        full_schedule = sorted([int(s) for s in (l1_schedule | {l1_total_steps} | l2_schedule_offset)])

        console.print(f"\n[bold]Generated Schedule for [cyan]{config.name}[/cyan]:[/bold]")
        yaml_dict = {"checkpoint_schedule": full_schedule}
        yaml_output = yaml.dump(yaml_dict, indent=2, default_flow_style=False)
        indented_yaml_output = "\n".join([f"    {line}" for line in yaml_output.splitlines()])
        console.print(indented_yaml_output)
        console.print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    app()