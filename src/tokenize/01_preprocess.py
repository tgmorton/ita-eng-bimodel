# 01_preprocess_reshaped.py
import os
import re
from glob import glob
from pathlib import Path

# The fixed number of words to put on each processed line.
WORDS_PER_LINE = 200

def reshape_file_content(text_content: str) -> list[str]:
    """
    Reshapes a single large block of text into lines with a fixed number of words.
    """
    # Split the entire text content into words using regex for any whitespace.
    words = re.split(r'\s+', text_content)
    # Filter out empty strings that result from multiple spaces.
    words = [word for word in words if word]

    final_lines = []
    # Group words into lines of a fixed length.
    for i in range(0, len(words), WORDS_PER_LINE):
        line = " ".join(words[i:i + WORDS_PER_LINE])
        final_lines.append(line)
    return final_lines


def main():
    """
    Finds all raw text files, converts them to lowercase, and reshapes them
    into lines of a fixed word count, saving the result to data/processed.
    """
    print("--- Running Script 01 (Reshaped): Preprocessing Raw Data ---")
    base_dir = Path.home() / "ita-eng-bimodel"
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    print(f"Reading raw from:    {raw_dir}")
    print(f"Writing processed to: {processed_dir}\n")

    # Find all .train and .test files to process
    all_files_to_process = glob(str(raw_dir / '**' / '*.train'), recursive=True)
    all_files_to_process += glob(str(raw_dir / '**' / '*.test'), recursive=True)


    if not all_files_to_process:
        print("FATAL ERROR: No .train or .test files found in the raw directory.")
        return

    print(f"Found {len(all_files_to_process)} total files to process.\n")

    for file_path_str in all_files_to_process:
        file_path = Path(file_path_str)
        relative_path = file_path.relative_to(raw_dir)
        output_path = processed_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  - Processing: {relative_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
            # Read the entire file content and convert it to lowercase.
            text_content = infile.read().lower()

        # Reshape the content into uniform lines.
        processed_lines = reshape_file_content(text_content)

        # Write the new, reshaped lines to the output file.
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in processed_lines:
                outfile.write(line + '\n')

    print(f"\n----- Preprocessing complete. All data reshaped to {WORDS_PER_LINE} words per line. -----")


if __name__ == '__main__':
    main()
