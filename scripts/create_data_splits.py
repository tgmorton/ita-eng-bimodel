# scripts/create_data_splits.py

import os
import re
import shutil
import collections
from pathlib import Path

# --- Configuration ---
# This dictionary holds the specific settings for each language.
# To create the subsets, words are proportionally removed ONLY from the
# files listed in `files_to_trim`. Other files are copied as-is.
DATA_CONFIG = {
    "italian": {
        "source_subfolder": "100M",
        "files_to_trim": [
            'PaCCSS-IT.train',
            'europarl-v7.it-en.it.train',
            'Leipzig_Web_Public.train',
            'SPGC.train',
            'QCRI.train'
        ],
        "targets": collections.OrderedDict([
            ("50M", 50_000_000),
            ("25M", 25_000_000),
            ("10M", 10_000_000),
        ])
    },
    "english": {
        "source_subfolder": "100M",
        # For English, we trim from all available files as they are all substantial.
        "files_to_trim": [
            'bnc_spoken.train',
            'childes.train',
            'gutenberg.train',
            'open_subtitles.train',
            'simple_wiki.train',
            'switchboard.train'
        ],
        "targets": collections.OrderedDict([
            ("50M", 50_000_000),
            ("25M", 25_000_000),
            ("10M", 10_000_000),
        ])
    }
}


# --- Helper Functions ---
def get_words_from_text(text_content: str) -> list:
    """Splits text into words using regex for whitespace and filters empty strings."""
    if not text_content:
        return []
    return [word for word in re.split(r'\s+', text_content) if word]


def write_words_to_file(filepath: Path, words_list: list):
    """Writes a list of words to a file, joined by single spaces."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(" ".join(words_list))


# --- Core Logic ---
def generate_derived_dataset(
        base_output_folder: Path,
        source_folder: Path,
        source_word_counts: dict,
        total_source_words: int,
        target_label: str,
        target_word_count: int,
        files_to_trim: list
):
    """Generates one smaller dataset from the source by trimming specified files."""
    print(f"\n--- Generating dataset: {target_label} (Target: {target_word_count:,} words) ---")
    target_output_folder = base_output_folder / target_label
    target_output_folder.mkdir(exist_ok=True)
    print(f"Output folder: {target_output_folder}")

    words_to_remove = total_source_words - target_word_count
    if words_to_remove <= 0:
        print("Source is smaller than target. Copying all files.")
        for filename in source_word_counts:
            shutil.copy2(source_folder / filename, target_output_folder / filename)
        return

    # Identify which of the specified files actually exist in the source
    operable_files = [f for f in files_to_trim if f in source_word_counts and source_word_counts[f] > 0]
    if not operable_files:
        print(f"Error: None of the specified trimming files exist in {source_folder}. Skipping.")
        return

    # Calculate total words available for trimming
    total_trimmable_words = sum(source_word_counts[f] for f in operable_files)
    print(f"Total words available for trimming in {len(operable_files)} specified files: {total_trimmable_words:,}")

    if total_trimmable_words < words_to_remove:
        print(
            f"Error: Not enough words ({total_trimmable_words:,}) in specified files to remove {words_to_remove:,}. Skipping.")
        return

    # Calculate how many words to cut from each operable file
    cuts_per_file = {}
    running_total_cut = 0
    if total_trimmable_words > 0:  # Avoid division by zero
        for filename in operable_files[:-1]:  # Process all but the last specified file
            proportion = source_word_counts[filename] / total_trimmable_words
            cut_amount = int(round(proportion * words_to_remove))
            cuts_per_file[filename] = cut_amount
            running_total_cut += cut_amount

    # The last file takes the remaining cut to ensure the total is exact
    if operable_files:
        last_file = operable_files[-1]
        cuts_per_file[last_file] = words_to_remove - running_total_cut

    # Now, create the new dataset
    final_word_count = 0
    for filename, source_wc in source_word_counts.items():
        source_path = source_folder / filename
        dest_path = target_output_folder / filename

        if filename in cuts_per_file:
            # This file needs to be trimmed
            num_to_cut = cuts_per_file[filename]
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                words = get_words_from_text(f.read())

            num_to_keep = max(0, len(words) - num_to_cut)
            trimmed_words = words[:num_to_keep]
            write_words_to_file(dest_path, trimmed_words)
            final_word_count += len(trimmed_words)
        else:
            # This file is not in the trim list, so copy it directly
            shutil.copy2(source_path, dest_path)
            final_word_count += source_wc

    print(f"Finished generating '{target_label}'. Final word count: {final_word_count:,}")


# --- Main Orchestration Script ---
def main_orchestrator(language: str, base_data_path: Path):
    """Creates smaller training datasets for the specified language."""

    config = DATA_CONFIG[language]
    source_folder_path = base_data_path / language / config["source_subfolder"]

    if not source_folder_path.is_dir():
        print(f"Error: Source directory '{source_folder_path}' not found.")
        return

    print(f"\n--- Analyzing Source: {source_folder_path} ---")
    source_word_counts = {}
    for filepath in sorted(source_folder_path.glob('*.train')):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source_word_counts[filepath.name] = len(get_words_from_text(f.read()))

    total_source_words = sum(source_word_counts.values())
    print(f"Source contains {total_source_words:,} words in {len(source_word_counts)} files.")

    # Generate each target dataset
    for target_label, target_wc in config["targets"].items():
        generate_derived_dataset(
            base_output_folder=base_data_path / language,
            source_folder=source_folder_path,
            source_word_counts=source_word_counts,
            total_source_words=total_source_words,
            target_label=target_label,
            target_word_count=target_wc,
            files_to_trim=config["files_to_trim"]
        )
    print(f"\n--- Finished processing for {language.capitalize()} ---")


def process_italian_test_data(base_data_path: Path):
    """Creates a smaller, 10M-word version of the Italian test dataset."""
    language = "italian"
    source_subfolder = "test_data"
    output_label = "test_data_reduced"
    target_wc = 10_000_000
    file_extension = ".test"

    source_folder_path = base_data_path / language / source_subfolder

    if not source_folder_path.is_dir():
        print(f"Error: Source test data directory '{source_folder_path}' not found.")
        return

    print(f"\n--- Processing Test Data: {source_folder_path} ---")

    # Analyze source test files
    source_word_counts = {}
    for filepath in sorted(source_folder_path.glob(f'*{file_extension}')):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source_word_counts[filepath.name] = len(get_words_from_text(f.read()))

    total_source_words = sum(source_word_counts.values())
    print(f"Source test data contains {total_source_words:,} words in {len(source_word_counts)} files.")

    # For the test set, we trim from all available files proportionally.
    files_to_trim = list(source_word_counts.keys())

    # Generate the reduced test dataset
    generate_derived_dataset(
        base_output_folder=base_data_path / language,
        source_folder=source_folder_path,
        source_word_counts=source_word_counts,
        total_source_words=total_source_words,
        target_label=output_label,
        target_word_count=target_wc,
        files_to_trim=files_to_trim
    )
    print(f"\n--- Finished processing Italian test data ---")


if __name__ == "__main__":
    print("This script creates smaller datasets (e.g., 50M, 25M, 10M) from a larger source.")

    # Hardcode the base path and process both languages automatically
    base_path = Path("~/ita-eng-bimodel/data/raw").expanduser()

    print(f"\nUsing base path: {base_path}")

    # Process training data
    main_orchestrator("english", base_path)
    main_orchestrator("italian", base_path)

    # Process Italian test data
    process_italian_test_data(base_path)

    print("\nAll dataset generation tasks complete.")
