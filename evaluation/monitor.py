# evaluation/monitor.py (Final Bilingual-Aware Version)

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from .model_wrapper import ModelWrapper
from .cases.surprisal_evaluation import SurprisalEvaluation
from .cases.perplexity_evaluation import PerplexityEvaluation
from .cases.priming_evaluation import PrimingEvaluation
from .data_loader import DataLoader as SurprisalDataLoader

EVAL_CASES_REGISTRY = {
    'surprisal': SurprisalEvaluation,
    'perplexity': PerplexityEvaluation,
    'priming': PrimingEvaluation,
}


def main():
    parser = argparse.ArgumentParser(description="Finds and evaluates model checkpoints locally.")
    parser.add_argument("--model_parent_dir", type=Path, required=True)
    parser.add_argument("--output_base_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument("--surprisal_data_dir", type=Path)
    parser.add_argument("--perplexity_data_base_path", type=Path)
    parser.add_argument("--priming_data_path", type=Path)
    parser.add_argument("--eval_cases", type=str, nargs='*', default=list(EVAL_CASES_REGISTRY.keys()))
    args = parser.parse_args()

    for checkpoint_path in sorted(list(args.model_parent_dir.glob("checkpoint-*"))):
        print(f"\n{'=' * 80}\nProcessing checkpoint: {checkpoint_path.name}")
        model_wrapper = ModelWrapper(checkpoint_path, args.tokenizer_path)
        output_dir = args.output_base_dir / args.model_parent_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for case_name in args.eval_cases:
            if case_name not in EVAL_CASES_REGISTRY: continue
            print(f"\n--- Running evaluation case: {case_name} ---")
            evaluator = EVAL_CASES_REGISTRY[case_name](model_wrapper)

            try:
                if case_name == 'surprisal':
                    if not args.surprisal_data_dir: raise ValueError("--surprisal_data_dir is required.")

                    # --- NEW BILINGUAL LOGIC ---
                    for csv_file in sorted(list(args.surprisal_data_dir.glob('*.csv'))):
                        temp_loader = SurprisalDataLoader(csv_file)
                        languages = temp_loader.get_available_languages()
                        if not languages:
                            print(f"  No processable languages found in {csv_file.name}, skipping.")
                            continue

                        for lang in languages:
                            eval_data = temp_loader.load_data(language=lang)
                            if eval_data.empty: continue

                            results = evaluator.run(eval_data, source_filename=csv_file.name)

                            # Save results per language
                            fname = f"{checkpoint_path.name}_{csv_file.stem}_{lang}_{case_name}.json"
                            with open(output_dir / fname, 'w') as f:
                                json.dump(results, f, indent=2)
                            print(f"Saving {lang} surprisal results to {output_dir / fname}")

                elif case_name == 'priming':
                    if not args.priming_data_path: raise ValueError("--priming_data_path is required.")
                    results = evaluator.run(data_path=args.priming_data_path)
                    fname = f"{checkpoint_path.name}_priming_results.json"
                    with open(output_dir / fname, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Saving priming results to {output_dir / fname}")

                # (Perplexity logic can be added here if needed)

            except Exception as e:
                print(f"  [ERROR] running case '{case_name}' for {checkpoint_path.name}: {e}")

    print(f"\n{'=' * 80}\nEvaluation monitor has completed.")


if __name__ == "__main__":
    main()