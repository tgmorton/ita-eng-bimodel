import pandas as pd
import json
from pathlib import Path
import argparse
import logging
import re

# Configure logging to provide clear feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def parse_filename(path: Path) -> dict:
    """
    Extracts metadata from the result filename. It now handles filenames
    with or without a language suffix.
    """
    # This regex is designed to be flexible.
    pattern = r'^(checkpoint-\d+)_([^_]+(?:_[^_]+)*?)(_(english|italian))?_surprisal\.json$'
    match = re.match(pattern, path.name)

    if match:
        return {
            "checkpoint_step": match.group(1),
            "source_file_stem": match.group(2),
            "language": match.group(4) if match.group(4) else 'unknown'
        }
    logging.warning(f"Could not parse metadata from filename: {path.name}")
    return {}


def process_surprisal_file(file_path: Path) -> pd.DataFrame:
    """
    Processes a single detailed surprisal JSON file and transforms it into a
    flat DataFrame suitable for analysis in R.
    """
    model_name = file_path.parent.name
    file_meta = parse_filename(file_path)
    if not file_meta:
        return pd.DataFrame()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
    except (json.JSONDecodeError, FileNotFoundError):
        return pd.DataFrame()

    output_rows = []
    for item in data:
        if not item or item.get("error"):
            continue

        null_hotspot = item.get('null_sentence_analysis', {}).get('hotspot_analysis', {})
        overt_hotspot = item.get('overt_sentence_analysis', {}).get('hotspot_analysis', {})

        row = {
            'model_run_name': model_name,
            'checkpoint_step': file_meta.get('checkpoint_step'),
            'language': file_meta.get('language'),
            'source_file': item.get('source_file'),
            'item_id': item.get('item_id'),
            'hotspot_text': item.get('hotspot_text'),
            'context': item.get('context'),
            'null_hotspot_avg_surprisal': null_hotspot.get('avg_surprisal'),
            'overt_hotspot_avg_surprisal': overt_hotspot.get('avg_surprisal'),
            'difference_score': item.get('hotspot_difference_score'),
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Collate bilingual surprisal evaluation JSON results into a single CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing model result folders (e.g., 'results/').")
    parser.add_argument("output_file", type=Path,
                        help="Path to save the final collated CSV file (e.g., 'analysis/surprisal_results.csv').")
    args = parser.parse_args()

    # Step 1: Find all possible surprisal result files
    all_json_files = sorted(list(args.results_dir.rglob("*_surprisal.json")))
    logging.info(f"Found {len(all_json_files)} total '*_surprisal.json' files.")

    # --- NEW: Filter files to only include those with language identifiers ---
    json_files_to_process = [
        f for f in all_json_files
        if '_english_' in f.name or '_italian_' in f.name
    ]
    logging.info(f"Filtered down to {len(json_files_to_process)} files containing 'english' or 'italian' to process.")

    if not json_files_to_process:
        logging.error("No relevant surprisal files found to process after filtering.")
        return

    # Step 2: Process each filtered file and concatenate the results
    all_dfs = [process_surprisal_file(f) for f in json_files_to_process]

    if not any(not df.empty for df in all_dfs):
        logging.warning("No valid data was extracted from any of the JSON files.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Step 3: Save the final collated data
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    logging.info(f"Successfully collated {len(final_df)} rows of data into: {args.output_file}")


if __name__ == "__main__":
    main()