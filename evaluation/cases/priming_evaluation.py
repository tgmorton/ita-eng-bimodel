# evaluation/cases/priming_evaluation.py

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper
from ..data_loader_priming import create_priming_dataloader  # Assuming the new loader is in a separate file

logger = logging.getLogger(__name__)


class PrimingEvaluation(EvaluationCase):
    """
    A self-contained evaluation case for calculating structural priming metrics.

    This case calculates:
    1. Sinclair-style priming effect (log-probability difference).
    2. Normalized probability of producing the congruent structure.
    3. The difference in normalized probabilities as a measure of priming.

    All metrics are calculated for both the full target sentence and the specific hotspot word.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _calculate_metrics_for_batch(self, batch: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
        """
        Calculates log-probabilities for the full target and the hotspot word
        for all 6 variations in a single batch.
        """
        device = self.model_wrapper.device
        tokenizer = self.model_wrapper.tokenizer
        use_amp = True  # Enable AMP for efficiency

        sequence_keys = [
            'con_prime_con_target_ids', 'con_prime_incon_target_ids',
            'incon_prime_con_target_ids', 'incon_prime_incon_target_ids',
            'base_con_target_ids', 'base_incon_target_ids'
        ]

        try:
            all_input_ids = torch.cat([batch[key].to(device) for key in sequence_keys], dim=0)
            all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long()
        except KeyError as e:
            logger.error(f"Batch from DataLoader is missing required tensor key: {e}")
            return {}

        original_batch_size = batch['con_prime_con_target_ids'].size(0)

        # Perform a single forward pass for all 6 sequence variations
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == 'cuda')):
            outputs = self.model_wrapper.model(input_ids=all_input_ids, attention_mask=all_attention_mask)
            logits = outputs.logits

        labels = all_input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        logits_list = torch.chunk(logits, len(sequence_keys), dim=0)
        labels_list = torch.chunk(labels, len(sequence_keys), dim=0)

        target_starts_map = {
            'logp_conT_conP': (batch['con_target_start_in_con_prime'], batch['con_target_hotspot_start'],
                               batch['con_target_hotspot_end']),
            'logp_inconT_conP': (batch['incon_target_start_in_con_prime'], batch['incon_target_hotspot_start'],
                                 batch['incon_target_hotspot_end']),
            'logp_conT_inconP': (batch['con_target_start_in_incon_prime'], batch['con_target_hotspot_start'],
                                 batch['con_target_hotspot_end']),
            'logp_inconT_inconP': (batch['incon_target_start_in_incon_prime'], batch['incon_target_hotspot_start'],
                                   batch['incon_target_hotspot_end']),
            'logp_conT_base': (torch.full((original_batch_size,), 1), batch['con_target_hotspot_start'],
                               batch['con_target_hotspot_end']),
            'logp_inconT_base': (torch.full((original_batch_size,), 1), batch['incon_target_hotspot_start'],
                                 batch['incon_target_hotspot_end']),
        }

        log_probs_per_item = [defaultdict(float) for _ in range(original_batch_size)]

        for key, (start_indices, hotspot_starts, hotspot_ends) in target_starts_map.items():
            logit_tensor, label_tensor = logits_list.pop(0), labels_list.pop(0)

            per_token_surprisals = -F.cross_entropy(
                logit_tensor.view(-1, logit_tensor.size(-1)),
                label_tensor.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(original_batch_size, -1)

            for i in range(original_batch_size):
                target_start_idx = start_indices[i].item()
                if target_start_idx == 0: continue

                item_labels = label_tensor[i, target_start_idx:]
                non_padding_len = (item_labels != -100).sum().item()
                if non_padding_len == 0: continue

                target_end_idx = target_start_idx + non_padding_len

                # Full sentence surprisal
                total_surprisal = per_token_surprisals[i, target_start_idx - 1: target_end_idx - 1].sum().item()
                log_probs_per_item[i][f'{key}_sentence'] = total_surprisal

                # Hotspot surprisal
                hotspot_start_in_target = hotspot_starts[i].item()
                hotspot_end_in_target = hotspot_ends[i].item()
                if hotspot_start_in_target != -1:
                    hotspot_start_in_seq = target_start_idx + hotspot_start_in_target - 1
                    hotspot_end_in_seq = target_start_idx + hotspot_end_in_target - 1
                    hotspot_surprisal = per_token_surprisals[i, hotspot_start_in_seq:hotspot_end_in_seq].sum().item()
                    log_probs_per_item[i][f'{key}_hotspot'] = hotspot_surprisal

        batch_results = defaultdict(list)
        for i in range(original_batch_size):
            batch_results[batch['target_structure'][i]].append(log_probs_per_item[i])

        return dict(batch_results)

    def _aggregate_metrics(self, results_list: List[Dict[str, float]], suffix: str) -> Dict[str, float]:
        """Aggregates raw log-probabilities into final priming metrics."""
        pe_scores, norm_p_con, norm_p_incon = [], [], []

        for probs in results_list:
            logp_ct_cp = probs.get(f'logp_conT_conP_{suffix}', float('nan'))
            logp_ct_ip = probs.get(f'logp_conT_inconP_{suffix}', float('nan'))
            logp_it_cp = probs.get(f'logp_inconT_conP_{suffix}', float('nan'))

            if math.isfinite(logp_ct_cp) and math.isfinite(logp_ct_ip):
                pe_scores.append(logp_ct_cp - logp_ct_ip)

            p_ct_cp, p_it_cp = math.exp(logp_ct_cp), math.exp(logp_it_cp)
            if p_ct_cp + p_it_cp > 0:
                norm_p_con.append(p_ct_cp / (p_ct_cp + p_it_cp))

            p_ct_ip = math.exp(logp_ct_ip)
            if p_ct_cp + p_ct_ip > 0:
                norm_p_incon.append(p_ct_ip / (p_ct_cp + p_ct_ip))

        avg_norm_p_con = np.mean(norm_p_con) if norm_p_con else float('nan')
        avg_norm_p_incon = np.mean(norm_p_incon) if norm_p_incon else float('nan')

        return {
            f'avg_pe_sinclair_{suffix}': np.mean(pe_scores) if pe_scores else float('nan'),
            f'priming_effect_normalized_{suffix}': avg_norm_p_con - avg_norm_p_incon,
        }

    def run(self, data_path: Path, batch_size: int = 4, **kwargs) -> List[Dict]:
        """
        Runs the priming evaluation on a directory of CSV files.

        Args:
            data_path (Path): Path to the directory containing priming data CSVs.
            batch_size (int): Batch size for evaluation.

        Returns:
            List[Dict]: A list of dictionaries with aggregated results per structure.
        """
        if not data_path.is_dir():
            raise FileNotFoundError(f"Priming data path must be a directory: {data_path}")

        csv_files = sorted(list(data_path.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        logger.info(f"Found {len(csv_files)} priming files to process.")

        all_raw_log_probs = defaultdict(list)
        for csv_file in csv_files:
            dataloader = create_priming_dataloader(
                csv_path=str(csv_file),
                tokenizer=self.model_wrapper.tokenizer,
                batch_size=batch_size
            )
            if not dataloader: continue

            for batch in tqdm(dataloader, desc=f"Evaluating {csv_file.name}", leave=False):
                batch_metrics = self._calculate_metrics_for_batch(batch)
                for structure, results in batch_metrics.items():
                    all_raw_log_probs[structure].extend(results)

        final_results = []
        for structure, results_list in all_raw_log_probs.items():
            sentence_metrics = self._aggregate_metrics(results_list, "sentence")
            hotspot_metrics = self._aggregate_metrics(results_list, "hotspot")

            structure_results = {"structure": structure, "count": len(results_list)}
            structure_results.update(sentence_metrics)
            structure_results.update(hotspot_metrics)
            final_results.append(structure_results)

        return final_results