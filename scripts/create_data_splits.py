# scripts/create_data_splits.py
import os
from glob import glob
from pathlib import Path


def count_words_in_file(file_path):
    """Counts the number of words in a file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return len(f.read().split())


def create_proportional_dataset(language, size_in_words, input_glob, output_dir):
    """
    Creates a dataset with a specific word limit by creating proportional
    versions of each source file.
    """
    print(f"--- Creating {language} dataset with approx {size_in_words:,} words (proportional, individual files) ---")
    files = sorted(glob(input_glob, recursive=True))
    if not files:
        print(f"  - WARNING: No files found for glob: {input_glob}")
        return

    file_word_counts = {f: count_words_in_file(f) for f in files}
    total_words = sum(file_word_counts.values())

    if total_words == 0:
        print("  - WARNING: No words found in source files. Skipping.")
        return

    if size_in_words > total_words:
        print(
            f"  - WARNING: Requested size ({size_in_words:,}) is larger than the total number of words available ({total_words:,}). Using all available words.")
        proportion = 1.0
    else:
        proportion = size_in_words / total_words

    output_dir.mkdir(parents=True, exist_ok=True)

    actual_total_words = 0

    for file_path, word_count in file_word_counts.items():
        words_to_take = int(word_count * proportion)
        actual_total_words += words_to_take

        original_filename = Path(file_path).name
        output_file_path = output_dir / original_filename

        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
            content = infile.read()
            words = content.split()

            with open(output_file_path, "w", encoding="utf-8") as outfile:
                outfile.write(" ".join(words[:words_to_take]))

    print(f"  - Successfully created proportional dataset with {actual_total_words:,} words.")
    print(f"  - Saved to: {output_dir}")


def main():
    """
    Main function to create all the required data splits.
    """
    base_dir = Path.home() / "ita-eng-bimodel"
    raw_dir = base_dir / "data" / "raw"

    # --- Create English Training Splits ---
    eng_sizes_in_words = [10_000_000, 25_000_000, 50_000_000]
    for size in eng_sizes_in_words:
        create_proportional_dataset(
            "english",
            size,
            str(raw_dir / "english" / "**" / "*.train"),
            raw_dir / "english" / f"{size // 1_000_000}M"
        )

    # --- Reduce Italian Test Set ---
    create_proportional_dataset(
        "italian",
        10_000_000,
        str(raw_dir / "italian" / "test_data" / "*.test"),
        raw_dir / "italian" / "test_data_reduced"
    )


if __name__ == "__main__":
    main()