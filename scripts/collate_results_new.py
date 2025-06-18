import pandas as pd
import json
from pathlib import Path
import argparse
import logging
from collections import defaultdict

# Configure logging to provide clear feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_priming_file(file_path: Path) -> pd.DataFrame:
    """
    Processes a single detailed priming JSON file and transforms it into a
    long-format DataFrame suitable for analysis in R.
    """
    model_name = file_path.parent.name
    checkpoint = file_path.name.split('_')[0]

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not data:
            logging.warning(f"File is empty, skipping: {file_path}")
            return pd.DataFrame()
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Could not read or parse {file_path}: {e}")
        return pd.DataFrame()

    # Step 1: Group trials by a unique item key (source file + item ID)
    grouped_items = defaultdict(list)
    for trial in data:
        meta = trial.get('metadata', {})
        key = (meta.get('source_file'), meta.get('item_id'))
        grouped_items[key].append(trial)

    output_rows = []

    # Step 2: Process each group of 2 trials (e.g., 'null' vs 'overt')
    for (source_file, item_id), trials in grouped_items.items():
        if len(trials) != 2:
            logging.warning(f"Skipping item {item_id} from {source_file} as it does not have a pair of trials.")
            continue

        # Ensure a consistent order for struct1 and struct2
        trial1 = trials[0] if trials[0]['metadata']['target_structure'] < trials[1]['metadata']['target_structure'] else \
        trials[1]
        trial2 = trials[1] if trials[0]['metadata']['target_structure'] < trials[1]['metadata']['target_structure'] else \
        trials[0]

        struct1_name = trial1['metadata']['target_structure']
        struct2_name = trial2['metadata']['target_structure']

        base_row = {
            'model_run_name': model_name,
            'checkpoint_step': checkpoint,
            'corpus_file': source_file,
            'item_id': item_id,
            'contrast_pair': f"{struct1_name}_vs_{struct2_name}",
        }

        # Get all metric names from one of the trials to ensure consistency
        all_metric_names = set(trial1.get('sentence_metrics', {}).keys()) | set(
            trial1.get('hotspot_metrics', {}).keys())

        for metric_name in sorted(list(all_metric_names)):
            # Handle sentence metrics
            if metric_name in trial1.get('sentence_metrics', {}):
                row = base_row.copy()
                row['metric_base'] = f"sentence_{metric_name}"
                row[f"value_{struct1_name}"] = trial1.get('sentence_metrics', {}).get(metric_name)
                row[f"value_{struct2_name}"] = trial2.get('sentence_metrics', {}).get(metric_name)
                output_rows.append(row)

            # Handle hotspot metrics
            if metric_name in trial1.get('hotspot_metrics', {}):
                row = base_row.copy()
                row['metric_base'] = f"hotspot_{metric_name}"
                row[f"value_{struct1_name}"] = trial1.get('hotspot_metrics', {}).get(metric_name)
                row[f"value_{struct2_name}"] = trial2.get('hotspot_metrics', {}).get(metric_name)
                output_rows.append(row)

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Collate detailed priming evaluation JSON results into a single, R-readable CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing model result folders (e.g., 'results/').")
    parser.add_argument("output_file", type=Path,
                        help="Path to save the final collated CSV file (e.g., 'analysis/priming_results.csv').")
    args = parser.parse_args()

    # Find all priming result files recursively
    json_files = sorted(list(args.results_dir.rglob("*_priming_results.json")))

    if not json_files:
        logging.error(f"No '*_priming_results.json' files found in {args.results_dir}. Please check the path.")
        return

    logging.info(f"Found {len(json_files)} priming result files to process...")

    # Process each file and concatenate the results
    all_dfs = [process_priming_file(f) for f in json_files]

    if not any(not df.empty for df in all_dfs):
        logging.warning("No valid data was extracted from any of the JSON files.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Create the output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    logging.info(f"Successfully collated {len(final_df)} rows of data into: {args.output_file}")


if __name__ == "__main__":
    main()