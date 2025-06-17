# count_raw_data.py
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table


def count_words_in_directory(directory_path: str, console: Console) -> int:
    """
    Counts the total number of words in all .train files within a given directory.
    """
    # Expand the user's home directory and create a Path object
    path = Path(directory_path).expanduser()

    if not path.is_dir():
        console.print(f"[bold red]Error: Directory not found at '{path}'[/bold red]")
        return 0

    total_words = 0
    # Use rglob to find all .train files, including in subdirectories
    train_files = list(path.rglob('*.train'))

    if not train_files:
        console.print(f"[bold red]Warning: No .train files found in '{path}'[/bold red]")
        return 0

    console.print(f"Found {len(train_files)} .train files in '{path}'. Counting words...")

    for file_path in train_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                words = content.split()
                total_words += len(words)
        except Exception as e:
            console.print(f"Could not read file {file_path}: {e}")

    return total_words


def main():
    """
    Main function to count words in the specified directory and display the result.
    """
    console = Console()

    # --- Configuration ---
    # The specific directory the user wants to count.
    target_directory = "~/ita-eng-bimodel/data/raw/english/100M"

    # --- Execution ---
    word_count = count_words_in_directory(target_directory, console)

    # --- Display Results ---
    table = Table(title="Raw Data Word Count")
    table.add_column("Directory Searched", style="cyan")
    table.add_column("Total Word Count", justify="right", style="bold green")

    table.add_row(target_directory, f"{word_count:,}")

    console.print(table)


if __name__ == "__main__":
    main()
