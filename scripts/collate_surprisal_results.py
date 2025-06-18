import pandas as pd
import json
from pathlib import Path
import argparse
import logging
import re

# Configure logging to provide clear feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def parse_filename(path: Path) -> dict:
    """Extracts metadata from the result filename."""
    # Regex to capture checkpoint, source file, and language
    # e.g., checkpoint-199_4a_subject_control_english_surprisal.json
    match = re.match(r'^(checkpoint-\d+)_([^_]+(?:_[^_]+)*)_(english|italian)_surprisal\.json$', path.name)
    if match:
        return {
            "checkpoint_step": match.group(1),
            "source_file": f"{match.group(2)}.csv",
            "language": match.group(3)
        }
    logging.warning(f"Could not parse metadata from filename: {path.name}")
    return {}


def process_surprisal_file(file_path: Path) -> pd.DataFrame:
    """
    Processes a single detailed surprisal JSON file and transforms it into a
    flat DataFrame suitable for analysis in R.
    """
    model_name = file_path.parent.name

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            logging.warning(f"File is empty, skipping: {file_path}")
            return pd.DataFrame()
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Could not read or parse {file_path}: {e}")
        return pd.DataFrame()

    output_rows = []

    for item in data:
        if not item or item.get("error"):
            continue

        # Extract metadata from the filename
        file_meta = parse_filename(file_path)

        # Extract surprisal values safely using .get() to avoid errors
        null_analysis = item.get('null_sentence_analysis', {})
        overt_analysis = item.get('overt_sentence_analysis', {})
        null_hotspot = null_analysis.get('hotspot_analysis', {})
        overt_hotspot = overt_analysis.get('hotspot_analysis', {})

        row = {
            'model_run_name': model_name,
            'checkpoint_step': file_meta.get('checkpoint_step'),
            'language': file_meta.get('language'),
            'source_file': item.get('source_file'),
            'item_id': item.get('item_id'),
            'hotspot_text': item.get('hotspot_text'),
            'context': item.get('context'),
            'null_hotspot_surprisal': null_hotspot.get('avg_surprisal'),
            'overt_hotspot_surprisal': overt_hotspot.get('avg_surprisal'),
            'difference_score': item.get('hotspot_difference_score'),
            'null_hotspot_tokens': " ".join(null_hotspot.get('tokens', [])).strip(),
            'overt_hotspot_tokens': " ".join(overt_hotspot.get('tokens', [])).strip(),
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Collate detailed surprisal evaluation JSON results into a single, R-readable CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing model result folders (e.g., 'results/').")
    parser.add_argument("output_file", type=Path,
                        help="Path to save the final collated CSV file (e.g., 'analysis/surprisal_results.csv').")
    args = parser.parse_args()

    # Find all surprisal result files recursively
    json_files = sorted(list(args.results_dir.rglob("*_surprisal.json")))

    if not json_files:
        logging.error(f"No '*_surprisal.json' files found in {args.results_dir}. Please check the path.")
        return

    logging.info(f"Found {len(json_files)} surprisal result files to process...")

    # Process each file and concatenate the results
    all_dfs = [process_surprisal_file(f) for f in json_files]

    if not any(not df.empty for df in all_dfs):
        logging.warning("No valid data was extracted from any of the JSON files.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure the output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    logging.info(f"Successfully collated {len(final_df)} rows of data into: {args.output_file}")


if __name__ == "__main__":
    main()