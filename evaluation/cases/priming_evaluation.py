# evaluation/cases/priming_evaluation.py (Final and Complete)

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

    def _calculate_metrics_for_batch(self, batch: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Calculates all surprisals for a batch and returns a list of per-item results.
        This method is now complete and robust.
        """
        if not batch: return []
        device = self.model_wrapper.device
        tokenizer = self.model_wrapper.tokenizer
        sequence_keys = ['con_prime_con_target_ids', 'con_prime_incon_target_ids', 'incon_prime_con_target_ids',
                         'incon_prime_incon_target_ids', 'base_con_target_ids', 'base_incon_target_ids']

        try:
            all_input_ids = torch.cat([batch[key].to(device) for key in sequence_keys], dim=0)
            all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long()
        except KeyError:
            return []

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
            logits = self.model_wrapper.model(input_ids=all_input_ids, attention_mask=all_attention_mask).logits

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
            'logp_conT_base': (torch.full((len(batch['metadata']),), 1), batch['con_target_hotspot_start'],
                               batch['con_target_hotspot_end']),
            'logp_inconT_base': (torch.full((len(batch['metadata']),), 1), batch['incon_target_hotspot_start'],
                                 batch['incon_target_hotspot_end'])}

        batch_results = [defaultdict(float) for _ in range(len(batch['metadata']))]
        for k, key in enumerate(target_starts_map.keys()):
            logit_tensor, label_tensor = logits_list[k], labels_list[k]
            start_indices, hotspot_starts, hotspot_ends = target_starts_map[key]

            surprisals = -F.cross_entropy(logit_tensor.view(-1, logit_tensor.size(-1)), label_tensor.view(-1),
                                          ignore_index=-100, reduction='none').view_as(label_tensor)

            for i in range(len(batch['metadata'])):
                target_start = start_indices[i].item()
                if target_start <= 0: continue
                target_len = (label_tensor[i, target_start:] != -100).sum().item()
                if target_len == 0: continue

                batch_results[i][f'{key}_sentence'] = surprisals[i,
                                                      target_start - 1: target_start - 1 + target_len].sum().item()

                h_start, h_end = hotspot_starts[i].item(), hotspot_ends[i].item()
                if h_start != -1:
                    h_start_seq = (target_start - 1) + h_start
                    h_end_seq = (target_start - 1) + h_end
                    batch_results[i][f'{key}_hotspot'] = surprisals[i, h_start_seq:h_end_seq].sum().item()
        return batch_results

    def _get_item_level_metrics(self, raw_probs: Dict, item_data: Dict, tokenized_data: Dict) -> Dict:
        """Calculates final metrics for a single item and returns a detailed dictionary."""
        metrics = {"metadata": item_data, "tokenization": tokenized_data, "sentence_metrics": {}, "hotspot_metrics": {}}
        for scope in ['sentence', 'hotspot']:
            logp_ct_cp = -raw_probs.get(f'logp_conT_conP_{scope}', float('nan'))
            logp_ct_ip = -raw_probs.get(f'logp_conT_inconP_{scope}', float('nan'))
            logp_it_cp = -raw_probs.get(f'logp_inconT_conP_{scope}', float('nan'))

            metrics[f"{scope}_metrics"] = {
                "pe_sinclair": logp_ct_cp - logp_ct_ip,
                "logp_congruent_target_given_congruent_prime": logp_ct_cp,
                "logp_incongruent_target_given_congruent_prime": logp_it_cp,
                "logp_congruent_target_given_incongruent_prime": logp_ct_ip,
                "norm_prob_congruent_target_given_congruent_prime": math.exp(logp_ct_cp) / (
                            math.exp(logp_ct_cp) + math.exp(logp_it_cp)) if (
                            logp_ct_cp != 0 and logp_it_cp != 0) else float('nan')
            }
        return metrics

    def run(self, data_path: Path, batch_size: int = 4, **kwargs) -> List[Dict]:
        all_results = []
        for csv_file in sorted(list(data_path.glob("*.csv"))):
            dataloader = create_priming_dataloader(csv_path=str(csv_file), tokenizer=self.model_wrapper.tokenizer,
                                                   batch_size=batch_size, num_workers=0)
            if not dataloader: continue

            for batch in tqdm(dataloader, desc=f"Evaluating {csv_file.name}", leave=False):
                item_metrics_list = self._calculate_metrics_for_batch(batch)
                if not item_metrics_list: continue

                for i, raw_probs in enumerate(item_metrics_list):
                    metadata = batch['metadata'][i]
                    tokenized_data = {k: self.model_wrapper.tokenizer.tokenize(v) for k, v in metadata.items() if
                                      isinstance(v, str)}
                    full_item_metrics = self._get_item_level_metrics(raw_probs, metadata, tokenized_data)
                    all_results.append(full_item_metrics)
        return all_results