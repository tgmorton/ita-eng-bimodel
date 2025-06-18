# evaluation/run_evaluation.py

import argparse
import json
from pathlib import Path

from evaluation.data_loader import DataLoader # For surprisal
from evaluation.model_wrapper import ModelWrapper
from evaluation.cases.surprisal_evaluation import SurprisalEvaluation
from evaluation.cases.perplexity_evaluation import PerplexityEvaluation
from evaluation.cases.priming_evaluation import PrimingEvaluation # <-- IMPORT NEW CASE

# A registry of all available evaluation cases
EVAL_CASES_REGISTRY = {
    'surprisal': SurprisalEvaluation,
    'perplexity': PerplexityEvaluation,
    'priming': PrimingEvaluation, # <-- ADD NEW CASE TO REGISTRY
}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation cases on a model checkpoint.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=Path, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the results JSON file.")

    parser.add_argument(
        "--eval_cases",
        type=str,
        nargs='*',
        default=list(EVAL_CASES_REGISTRY.keys()),
        help=f"Space-separated list of evaluation cases. Available: {list(EVAL_CASES_REGISTRY.keys())}."
    )

    # Add specific data path arguments for each case
    parser.add_argument("--surprisal_data_path", type=Path, help="Path to the CSV file for surprisal evaluation.")
    parser.add_argument("--perplexity_data_path", type=Path, help="Path to tokenized dataset for perplexity.")
    parser.add_argument("--priming_data_path", type=Path, help="Path to the DIRECTORY with CSV files for priming.") # <-- NEW ARGUMENT
    parser.add_argument("--max_samples", type=int, default=None, help="For perplexity, max number of samples.")

    args = parser.parse_args()

    print(f"Loading model and tokenizer for checkpoint: {args.checkpoint_path.name}...")
    model_wrapper = ModelWrapper(args.checkpoint_path, args.tokenizer_path)

    for case_name in args.eval_cases:
        evaluator_class = EVAL_CASES_REGISTRY.get(case_name)
        if not evaluator_class:
            print(f"Warning: Unknown evaluation case '{case_name}'. Skipping.")
            continue

        print(f"\n--- Running evaluation case: {case_name} ---")
        evaluator = evaluator_class(model_wrapper)
        results = []
        data_name_for_filename = "unknown"

        try:
            if case_name == 'surprisal':
                if not args.surprisal_data_path:
                    raise ValueError("Surprisal requires --surprisal_data_path.")
                data_loader = DataLoader(args.surprisal_data_path)
                eval_data = data_loader.load_data()
                results = evaluator.run(eval_data)
                data_name_for_filename = args.surprisal_data_path.stem

            elif case_name == 'perplexity':
                if not args.perplexity_data_path:
                    raise ValueError("Perplexity requires --perplexity_data_path.")
                results = evaluator.run(data_path=args.perplexity_data_path, max_samples=args.max_samples)
                data_name_for_filename = args.perplexity_data_path.name

            # --- ADD LOGIC FOR THE NEW PRIMING CASE ---
            elif case_name == 'priming':
                if not args.priming_data_path:
                    raise ValueError("Priming evaluation requires --priming_data_path.")
                results = evaluator.run(data_path=args.priming_data_path)
                data_name_for_filename = "priming_results"
            # -----------------------------------------

            # Save results for the current case
            output_filename = f"{args.checkpoint_path.name}_{data_name_for_filename}_{case_name}.json"
            output_path = args.output_dir / output_filename
            args.output_dir.mkdir(exist_ok=True, parents=True)

            print(f"Saving {case_name} results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error running evaluation case '{case_name}': {e}", exc_info=True)
            continue

    print("\nAll specified evaluations complete for this checkpoint.")


if __name__ == "__main__":
    main()