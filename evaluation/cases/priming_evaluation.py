# evaluation/cases/priming_evaluation.py (Enhanced for Item-Level Logging)

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper
from ..data_loader_priming import create_priming_dataloader

logger = logging.getLogger(__name__)


class PrimingEvaluation(EvaluationCase):
    """
    A self-contained evaluation case for calculating structural priming metrics.
    This version provides detailed, item-level output for every trial.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _get_item_level_metrics(self, raw_probs: Dict[str, float], item_data: Dict, tokenized_data: Dict) -> Dict:
        """Calculates all metrics for a single item and returns a detailed dictionary."""
        metrics = {"metadata": item_data}

        # --- Tokenization Info ---
        metrics["tokenization"] = {
            "congruent_prime": tokenized_data["congruent_prime"],
            "incongruent_prime": tokenized_data["incongruent_prime"],
            "congruent_target": tokenized_data["congruent_target"],
            "incongruent_target": tokenized_data["incongruent_target"],
            "hotspot": tokenized_data["hotspot"],
        }

        # --- Sentence-level metrics ---
        logp_ct_cp_s = raw_probs.get('logp_conT_conP_sentence', float('nan'))
        logp_it_cp_s = raw_probs.get('logp_inconT_conP_sentence', float('nan'))
        logp_ct_ip_s = raw_probs.get('logp_conT_inconP_sentence', float('nan'))

        metrics["sentence_metrics"] = {
            "pe_sinclair": logp_ct_cp_s - logp_ct_ip_s,
            "logp_congruent_target_given_congruent_prime": logp_ct_cp_s,
            "logp_incongruent_target_given_congruent_prime": logp_it_cp_s,
            "logp_congruent_target_given_incongruent_prime": logp_ct_ip_s,
            "norm_prob_congruent_target_given_congruent_prime": math.exp(logp_ct_cp_s) / (
                        math.exp(logp_ct_cp_s) + math.exp(logp_it_cp_s))
        }

        # --- Hotspot-level metrics ---
        logp_ct_cp_h = raw_probs.get('logp_conT_conP_hotspot', float('nan'))
        logp_it_cp_h = raw_probs.get('logp_inconT_conP_hotspot', float('nan'))
        logp_ct_ip_h = raw_probs.get('logp_conT_inconP_hotspot', float('nan'))

        metrics["hotspot_metrics"] = {
            "pe_sinclair": logp_ct_cp_h - logp_ct_ip_h,
            "logp_congruent_target_given_congruent_prime": logp_ct_cp_h,
            "logp_incongruent_target_given_congruent_prime": logp_it_cp_h,
            "logp_congruent_target_given_incongruent_prime": logp_ct_ip_h,
            "norm_prob_congruent_target_given_congruent_prime": math.exp(logp_ct_cp_h) / (
                        math.exp(logp_ct_cp_h) + math.exp(logp_it_cp_h))
        }

        return metrics

    def run(self, data_path: Path, batch_size: int = 4, **kwargs) -> List[Dict]:
        """
        Runs priming evaluation and returns a list of detailed, item-level dictionaries.
        """
        # (The _calculate_metrics_for_batch method remains the same as the last version)
        # ...

        all_results = []
        for csv_file in sorted(list(data_path.glob("*.csv"))):
            dataloader = create_priming_dataloader(csv_path=str(csv_file), tokenizer=self.model_wrapper.tokenizer,
                                                   batch_size=batch_size)
            if not dataloader: continue

            # We need the original items to map results back
            original_items = dataloader.dataset.data
            item_iterator = iter(original_items)

            for batch in tqdm(dataloader, desc=f"Evaluating {csv_file.name}", leave=False):
                raw_results_by_structure = self._calculate_metrics_for_batch(batch)

                # Reconstruct item-level data
                for i in range(batch_size):
                    try:
                        item_data = next(item_iterator)
                        structure = item_data['target_structure']
                        # Find the corresponding result for this item in the batch output
                        if structure in raw_results_by_structure and raw_results_by_structure[structure]:
                            raw_probs = raw_results_by_structure[structure].pop(0)

                            # Tokenize for logging purposes
                            tokenized_data = {k: self.model_wrapper.tokenizer.tokenize(v) for k, v in item_data.items()
                                              if isinstance(v, str)}

                            # Get all metrics for this single item
                            full_item_metrics = self._get_item_level_metrics(raw_probs, item_data, tokenized_data)
                            all_results.append(full_item_metrics)

                    except StopIteration:
                        break  # End of batch

        return all_results

    # Paste the _calculate_metrics_for_batch method from the previous response here.
    # It does not need to be changed.
    def _calculate_metrics_for_batch(self, batch: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
        # ... (Method from previous response)
        pass  # Placeholder