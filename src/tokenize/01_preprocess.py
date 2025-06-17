# 01_preprocess.py
import os
import nltk
from glob import glob

CHARACTER_LIMIT = 4000


def smart_split_text(text, language='italian'):
    """
    Splits text using a hybrid strategy.
    1. First, attempts to split by sentences using NLTK.
    2. Then, for any resulting sentence that is still too long, it splits that
       sentence at the nearest word boundary under the CHARACTER_LIMIT.
    """
    final_lines = []
    initial_sentences = nltk.sent_tokenize(text, language=language)
    for sentence in initial_sentences:
        if len(sentence.encode('utf-8')) <= CHARACTER_LIMIT:
            final_lines.append(sentence)
        else:
            current_pos = 0
            while current_pos < len(sentence):
                end_pos = current_pos + CHARACTER_LIMIT
                if end_pos >= len(sentence):
                    chunk = sentence[current_pos:]
                    final_lines.append(chunk)
                    break
                split_pos = sentence.rfind(' ', current_pos, end_pos)
                if split_pos != -1:
                    chunk = sentence[current_pos:split_pos]
                    current_pos = split_pos + 1
                else:
                    chunk = sentence[current_pos:end_pos]
                    current_pos = end_pos
                final_lines.append(chunk)
    return final_lines


def main():
    """
    Finds all raw text, converts it to lowercase, splits it into clean sentences,
    and saves the result to data/processed.
    """
    print("--- Running Script 01: Preprocessing Raw Data ---")
    base_dir = os.path.expanduser('~/ita-eng-bimodel')
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    print(f"Reading raw from:    {raw_dir}")
    print(f"Writing processed to: {processed_dir}\n")

    train_files_to_process = glob(os.path.join(raw_dir, '**', '*.train'), recursive=True)
    test_files_to_process = glob(os.path.join(raw_dir, '**', '*.test'), recursive=True)
    all_files_to_process = train_files_to_process + test_files_to_process

    if not all_files_to_process:
        print("FATAL ERROR: No .train or .test files found in the raw directory.")
        return

    print(f"Found {len(train_files_to_process)} .train files.")
    print(f"Found {len(test_files_to_process)} .test files.")
    print(f"Total files to process: {len(all_files_to_process)}\n")

    for file_path in all_files_to_process:
        relative_path = os.path.relpath(file_path, raw_dir)
        output_path = os.path.join(processed_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"  - Processing: {relative_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            # Read the entire file content and convert it to lowercase
            text_content = infile.read().lower()  # <<< THIS LINE IS MODIFIED

            processed_lines = smart_split_text(text_content)
            for line in processed_lines:
                cleaned_line = line.strip()
                if cleaned_line:
                    outfile.write(cleaned_line + '\n')

    print("\n----- Preprocessing complete. All text is now lowercase. -----")


if __name__ == '__main__':
    main()