# analysis/collate_results.py (Final and Complete)

import pandas as pd
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def flatten_surprisal_data(data: list, checkpoint: str, model_name: str) -> pd.DataFrame:
    """Flattens the detailed surprisal JSON into a long-format DataFrame."""
    rows = []
    for item in data:
        if not item: continue
        row_base = {
            'model_name': model_name,
            'checkpoint_step': checkpoint,
            'source_file': item.get('source_file'),
            'item_id': item.get('item_id')
        }
        rows.append({**row_base, 'metric_type': 'surprisal', 'metric_name': 'hotspot_difference_score',
                     'value': item.get('hotspot_difference_score')})

        # You can extract more details here if needed, e.g., avg surprisal for null/overt
        for analysis_type in ['null', 'overt']:
            hotspot_analysis = item.get(f'{analysis_type}_sentence_analysis', {}).get('hotspot_analysis', {})
            rows.append({**row_base, 'metric_type': 'surprisal', 'metric_name': f'avg_surprisal_{analysis_type}',
                         'value': hotspot_analysis.get('avg_surprisal')})

    return pd.DataFrame([r for r in rows if r.get('value') is not None])


def flatten_priming_data(data: list, checkpoint: str, model_name: str) -> pd.DataFrame:
    """Flattens the detailed priming JSON into the desired long-format DataFrame."""
    rows = []
    for item in data:
        if not item: continue
        meta = item.get('metadata', {})
        base_row = {
            'model_name': model_name,
            'checkpoint_step': checkpoint,
            'source_file': meta.get('source_file'),
            'item_id': meta.get('item_id'),
            'structure': meta.get('target_structure'),
        }

        for scope in ['sentence', 'hotspot']:
            metrics = item.get(f'{scope}_metrics', {})
            for metric, value in metrics.items():
                rows.append({**base_row, 'metric_type': scope, 'metric_name': metric, 'value': value})

    return pd.DataFrame([r for r in rows if r.get('value') is not None])


def main():
    parser = argparse.ArgumentParser(description="Collate detailed evaluation JSON results into a single CSV.")
    parser.add_argument("results_dir", type=Path, help="Directory containing model result folders (e.g., 'results/').")
    parser.add_argument("output_file", type=Path, help="Path to save the final collated CSV file.")
    args = parser.parse_args()

    all_dfs = []
    json_files = sorted(list(args.results_dir.rglob("*.json")))

    if not json_files:
        logging.warning(f"No .json files found in {args.results_dir}. Exiting.")
        return

    logging.info(f"Found {len(json_files)} result files to process...")

    for file_path in json_files:
        model_name = file_path.parent.name
        checkpoint = file_path.name.split('_')[0]

        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if not data:
                    logging.warning(f"File {file_path} is empty. Skipping.")
                    continue
            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from {file_path}. Skipping.")
                continue

        if "_surprisal.json" in file_path.name:
            all_dfs.append(flatten_surprisal_data(data, checkpoint, model_name))
        elif "_priming.json" in file_path.name:
            all_dfs.append(flatten_priming_data(data, checkpoint, model_name))

    if not all_dfs:
        logging.info("No valid data was extracted from the JSON files.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['value'])
    final_df.to_csv(args.output_file, index=False)
    logging.info(f"Successfully collated {len(final_df)} rows of data into: {args.output_file}")


if __name__ == "__main__":
    main()