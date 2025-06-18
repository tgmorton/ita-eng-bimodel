# evaluation/monitor.py

import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Import all evaluation components
from .model_wrapper import ModelWrapper
from .cases.surprisal_evaluation import SurprisalEvaluation
from .cases.perplexity_evaluation import PerplexityEvaluation
# --- NEW: Import the Priming Evaluation Case ---
from .cases.priming_evaluation import PrimingEvaluation
from .data_loader import DataLoader as SurprisalDataLoader

# --- NEW: Updated Registry with Priming ---
EVAL_CASES_REGISTRY = {
    'surprisal': SurprisalEvaluation,
    'perplexity': PerplexityEvaluation,
    'priming': PrimingEvaluation,
}


def main():
    parser = argparse.ArgumentParser(description="Finds and evaluates model checkpoints locally.")

    # --- MODIFIED: Simplified core paths ---
    parser.add_argument("--model_parent_dir", type=Path, required=True,
                        help="Path to the model directory (e.g., /path/to/models/10M_10epoch).")
    parser.add_argument("--output_base_dir", type=Path, required=True,
                        help="Base directory to save all evaluation results.")
    # --- MODIFIED: Changed to a single, direct tokenizer path ---
    parser.add_argument("--tokenizer_path", type=Path, required=True, help="Direct path to the tokenizer directory.")

    # --- Data paths for specific evaluation cases ---
    parser.add_argument("--surprisal_data_dir", type=Path,
                        help="Path to the directory with CSV files for surprisal evaluation.")
    parser.add_argument("--perplexity_data_base_path", type=Path,
                        help="Path to the base dir for tokenized data (e.g., /data/tokenized).")
    # --- NEW: Argument for priming data ---
    parser.add_argument("--priming_data_path", type=Path,
                        help="Path to the DIRECTORY with CSV files for priming evaluation.")

    # --- General options ---
    parser.add_argument("--perplexity_eval_portion", type=float, default=1.0,
                        help="Portion of the perplexity test set to use (default: 1.0).")
    parser.add_argument(
        "--eval_cases", type=str, nargs='*', default=list(EVAL_CASES_REGISTRY.keys()),
        help=f"Space-separated list of cases to run. Default: all. Available: {list(EVAL_CASES_REGISTRY.keys())}"
    )
    args = parser.parse_args()

    print(f"--- Starting Local Evaluation Monitor ---")
    print(f"Searching for model checkpoints in: {args.model_parent_dir}")

    all_checkpoints = sorted(list(args.model_parent_dir.glob("checkpoint-*")))
    if not all_checkpoints:
        print(f"No checkpoints found in {args.model_parent_dir}. Exiting.")
        return

    print(f"Found {len(all_checkpoints)} total checkpoints to evaluate.")
    print(f"Using Tokenizer: {args.tokenizer_path}")

    for checkpoint_path in all_checkpoints:
        try:
            print(f"\n{'=' * 80}")
            print(f"Processing checkpoint: {checkpoint_path.name}")

            # --- MODIFIED: Simplified model loading ---
            model_wrapper = ModelWrapper(checkpoint_path, args.tokenizer_path)
            output_dir = args.output_base_dir / args.model_parent_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)

            for case_name in args.eval_cases:
                if case_name not in EVAL_CASES_REGISTRY:
                    print(f"  [Warning] Unknown eval case '{case_name}'. Skipping.")
                    continue

                evaluator_class = EVAL_CASES_REGISTRY[case_name]
                print(f"\n--- Running evaluation case: {case_name} ---")
                evaluator = evaluator_class(model_wrapper)
                results = []
                data_name_for_filename = "results"

                try:
                    # --- Dispatch to the correct evaluation case ---
                    if case_name == 'surprisal':
                        if not args.surprisal_data_dir: raise ValueError("--surprisal_data_dir is required.")
                        # (Logic for running surprisal on multiple files remains)
                        all_surprisal_results = []
                        for csv_file in sorted(list(args.surprisal_data_dir.glob('*.csv'))):
                            data_loader = SurprisalDataLoader(csv_file)
                            eval_data = data_loader.load_data()
                            all_surprisal_results.extend(evaluator.run(eval_data, source_filename=csv_file.name))
                        results = all_surprisal_results
                        data_name_for_filename = "surprisal_stimuli"

                    elif case_name == 'perplexity':
                        if not args.perplexity_data_base_path: raise ValueError(
                            "--perplexity_data_base_path is required.")
                        # Assumes the test set is named 'english_test' or 'italian_test' inside the tokenizer dir
                        model_name = args.tokenizer_path.name
                        perplexity_data_path = args.perplexity_data_base_path / model_name / "english_test"
                        if not perplexity_data_path.exists():
                            perplexity_data_path = args.perplexity_data_base_path / model_name / "italian_test"
                        if not perplexity_data_path.exists():
                            raise FileNotFoundError(f"Could not find a test set for {model_name}")

                        results = evaluator.run(data_path=perplexity_data_path,
                                                eval_portion=args.perplexity_eval_portion)
                        data_name_for_filename = f"{perplexity_data_path.parent.name}_{perplexity_data_path.name}"

                    # --- NEW: Logic to run the Priming Evaluation Case ---
                    elif case_name == 'priming':
                        if not args.priming_data_path: raise ValueError("--priming_data_path is required.")
                        results = evaluator.run(data_path=args.priming_data_path)
                        data_name_for_filename = "priming"

                    # Save results for the current case
                    output_filename = f"{checkpoint_path.name}_{data_name_for_filename}_{case_name}.json"
                    output_path = output_dir / output_filename
                    print(f"Saving {case_name} results to {output_path}...")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"  [ERROR] running case '{case_name}' for {checkpoint_path.name}: {e}")
                    continue

        except Exception as e:
            print(f"  [FATAL ERROR] processing checkpoint {checkpoint_path.name}. Error: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("Evaluation monitor has completed.")


if __name__ == "__main__":
    main()