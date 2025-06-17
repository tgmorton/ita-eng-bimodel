# count_words.py
import os
import glob
from pathlib import Path
from rich.console import Console
from rich.table import Table


def count_words_in_dir(dir_path: Path, console: Console) -> int:
    """Counts the total words in all .train files in a directory."""
    total_words = 0
    # Use rglob to find .train files in subdirectories as well
    files = list(dir_path.rglob('*.train'))

    if not files:
        console.print(f"[bold red]Warning: No .train files found in {dir_path}[/bold red]")
        return 0

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_words += len(f.read().split())

    return total_words


def main():
    """
    Counts the actual words in the split datasets to verify their sizes.
    """
    console = Console()
    base_dir = Path.home() / "ita-eng-bimodel" / "data" / "raw"

    table = Table(title="Final Word Count Verification")
    table.add_column("Dataset", style="cyan")
    table.add_column("Path Searched", style="dim")
    table.add_column("Actual Word Count", justify="right", style="bold green")

    # --- Check a 25M configuration ---
    eng_path = base_dir / "english" / "25M"
    ita_path = base_dir / "italian" / "25M"

    eng_words = count_words_in_dir(eng_path, console)
    ita_words = count_words_in_dir(ita_path, console)

    table.add_row("English 25M (Target)", str(eng_path), f"{eng_words:,}")
    table.add_row("Italian 25M (Target)", str(ita_path), f"{ita_words:,}")

    console.print(table)


if __name__ == "__main__":
    main()