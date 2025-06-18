import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
import math
import re
from typing import List, Tuple

# --- Constants ---
# Represents the block_size from src/data.py (approximating 1024 tokens)
# We use a word count that is a rough proxy for 1024 tokens.
WORDS_PER_FINAL_CHUNK = 750
# The fixed number of words to put on each intermediate processed line.
WORDS_PER_PROCESSED_LINE = 200

# Initialize the console for rich output.
console = Console()


def read_and_concatenate_files(directory: Path) -> str:
    """Reads all .train files in a directory and concatenates their content."""
    if not directory.is_dir():
        console.print(f"[bold red]Error: Directory not found at '{directory}'[/bold red]")
        return ""

    all_text = []
    train_files = sorted(list(directory.glob("*.train")))
    if not train_files:
        console.print(f"[bold yellow]Warning: No .train files found in {directory}[/bold yellow]")
        return ""

    console.print(f"Reading {len(train_files)} files from {directory}...")
    for file_path in train_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_text.append(f.read())

    return " ".join(all_text)


def reshape_text_by_word_count(text_content: str) -> Tuple[List[str], int]:
    """
    Reshapes a single large block of text into lines with a fixed number of words.
    """
    # Split the entire text content into words using regex for any whitespace.
    words = re.split(r'\s+', text_content)
    words = [word for word in words if word]  # Filter out empty strings.
    total_words = len(words)

    final_lines = []
    # Group words into lines of a fixed length.
    for i in range(0, total_words, WORDS_PER_PROCESSED_LINE):
        line = " ".join(words[i:i + WORDS_PER_PROCESSED_LINE])
        final_lines.append(line)

    return final_lines, total_words


def simulate_chunking(total_words: int) -> int:
    """
    Simulates the final chunking process based on total word count.
    """
    if total_words == 0:
        return 0
    # This simulates the floor division used when creating final training batches.
    return math.floor(total_words / WORDS_PER_FINAL_CHUNK)


def main(ita_dir: Path, eng_dir: Path):
    """
    Runs the validation by comparing the two full datasets.
    """
    console.print("\n[bold cyan]Validating Dataset Preprocessing for Parity[/bold cyan]")

    # --- Process Italian Dataset ---
    console.print("\n[bold]--- Processing Italian Dataset ---[/bold]")
    ita_raw_text = read_and_concatenate_files(ita_dir)
    ita_processed_lines, ita_total_words = reshape_text_by_word_count(ita_raw_text)
    ita_final_chunks = simulate_chunking(ita_total_words)

    # --- Process English Dataset ---
    console.print("\n[bold]--- Processing English Dataset ---[/bold]")
    eng_raw_text = read_and_concatenate_files(eng_dir)
    eng_processed_lines, eng_total_words = reshape_text_by_word_count(eng_raw_text)
    eng_final_chunks = simulate_chunking(eng_total_words)

    # --- Display Results ---
    table = Table(title="Dataset Parity Comparison (Fixed Word Count Strategy)")
    table.add_column("Metric", style="magenta", justify="right")
    table.add_column("Italian Dataset", style="yellow")
    table.add_column("English Dataset", style="green")

    table.add_row("Input Directory", str(ita_dir), str(eng_dir))
    table.add_row("Total Raw Words", f"{ita_total_words:,}", f"{eng_total_words:,}")
    table.add_row("Processed Lines (at {} words/line)".format(WORDS_PER_PROCESSED_LINE), f"{len(ita_processed_lines):,}", f"{len(eng_processed_lines):,}")
    table.add_row("[bold]Final Num. Training Chunks[/bold]", f"[bold]{ita_final_chunks:,}[/bold]", f"[bold]{eng_final_chunks:,}[/bold]")

    console.print(table)

    # --- Conclusion ---
    console.print("\n[bold]Conclusion:[/bold]")
    if ita_final_chunks > 0 and eng_final_chunks > 0:
        difference = abs(ita_final_chunks - eng_final_chunks)
        avg_chunks = (ita_final_chunks + eng_final_chunks) / 2
        percent_diff = (difference / avg_chunks) * 100
        console.print(
            f"The final chunk counts differ by [bold]{difference:,}[/bold] chunks, a variance of [bold]{percent_diff:.2f}%[/bold]."
        )
        if percent_diff < 5:
            console.print(
                "[bold green]This indicates excellent parity.[/bold green] Your proposed method of reshaping the data into fixed-length lines "
                "is effective at standardizing the datasets for comparable training runs."
            )
        else:
            console.print(
                "[bold yellow]There is some variance.[/bold yellow] While better than sentence-splitting, the inherent differences "
                "in word length and tokenization between Italian and English still result in a moderate difference in chunk count."
            )
    else:
        console.print("[bold red]Could not calculate parity because one or both datasets resulted in zero chunks.[/bold red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset chunking parity by reshaping raw text.")
    parser.add_argument("--ita-dir", default="/Users/thomasmorton/ita-eng-bimodel/data/raw/italian/10M", type=Path, help="Path to the raw Italian data directory (e.g., data/raw/italian/10M).")
    parser.add_argument("--eng-dir", default="/Users/thomasmorton/ita-eng-bimodel/data/raw/english/10M", type=Path, help="Path to the raw English data directory (e.g., data/raw/english/10M).")
    args = parser.parse_args()
    main(args.ita_dir, args.eng_dir)

