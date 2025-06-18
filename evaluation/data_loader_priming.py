# evaluation/data_loader_priming.py (Final and Complete)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PrimingEvaluationDataset(Dataset):
    def __init__(self, processed_data: List[Dict[str, Any]]):
        self.data = processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def find_subsequence_indices(sequence: List[int], subsequence: List[int]) -> Tuple[int, int]:
    sub_len = len(subsequence)
    if sub_len == 0: return -1, -1
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            return i, i + sub_len
    logger.warning(f"Hotspot subsequence not found in target sequence.")
    return -1, -1


def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    prime_structures = set()
    for col in columns:
        col = col.strip()
        if col.startswith('p_'):
            structure_name = col.split('_', 1)[1]
            if structure_name:
                prime_structures.add(structure_name)
    if len(prime_structures) == 2:
        return tuple(sorted(list(prime_structures)))
    logger.warning(f"Could not determine exactly two structures from prime columns. Found: {prime_structures}")
    return None


def load_and_process_priming_data(csv_path: Path) -> List[Dict[str, Any]]:
    processed_data = []
    try:
        df = pd.read_csv(csv_path).fillna('')
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error loading CSV {csv_path.name}: {e}");
        return []

    alternation = get_structure_alternations(list(df.columns))
    if alternation is None: return []

    struct1, struct2 = alternation
    p_col1, p_col2 = f'p_{struct1}', f'p_{struct2}'
    t_col1, t_col2 = f't_{struct1}', f't_{struct2}'
    c_p_col, c_t_col, hotspot_t_col = 'c_p', 'c_t', 'hotspot_t'

    required_cols = [p_col1, p_col2, t_col1, t_col2, c_p_col, c_t_col, hotspot_t_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"CSV {csv_path.name} missing columns: {list(set(required_cols) - set(df.columns))}")
        return []

    for index, row in df.iterrows():
        try:
            item_base = {'item_id': str(row.get('item', index)), 'source_file': csv_path.name}
            prime_context, target_context = str(row[c_p_col]), str(row[c_t_col])
            hotspot, prime1, prime2 = str(row[hotspot_t_col]), str(row[p_col1]), str(row[p_col2])
            target1, target2 = str(row[t_col1]), str(row[t_col2])

            # Create two evaluation items per row from the CSV
            processed_data.append({
                **item_base, 'target_structure': struct1, 'hotspot': hotspot,
                'congruent_prime': f"{prime_context} {prime1}".strip(),
                'incongruent_prime': f"{prime_context} {prime2}".strip(),
                'congruent_target': f"{target_context} {target1}".strip(),
                'incongruent_target': f"{target_context} {target2}".strip(),
            })
            processed_data.append({
                **item_base, 'target_structure': struct2, 'hotspot': hotspot,
                'congruent_prime': f"{prime_context} {prime2}".strip(),
                'incongruent_prime': f"{prime_context} {prime1}".strip(),
                'congruent_target': f"{target_context} {target2}".strip(),
                'incongruent_target': f"{target_context} {target1}".strip(),
            })
        except Exception:
            continue
    return processed_data


def collate_for_priming_eval(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    if not batch: return {}
    collated = defaultdict(list)
    bos_id, pad_id = tokenizer.bos_token_id, tokenizer.pad_token_id
    if bos_id is None or pad_id is None: return {}

    tokenized = defaultdict(list)
    for item in batch:
        for key in ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target', 'hotspot']:
            tokenized[key].append(tokenizer(item[key], add_special_tokens=False)['input_ids'])

    unpadded_sequences, global_max_len = defaultdict(list), 0
    for i in range(len(batch)):
        cp_toks, ip_toks = tokenized['congruent_prime'][i], tokenized['incongruent_prime'][i]
        ct_toks, it_toks = tokenized['congruent_target'][i], tokenized['incongruent_target'][i]

        variants = {
            'con_prime_con_target_ids': [bos_id] + cp_toks + ct_toks,
            'con_prime_incon_target_ids': [bos_id] + cp_toks + it_toks,
            'incon_prime_con_target_ids': [bos_id] + ip_toks + ct_toks,
            'incon_prime_incon_target_ids': [bos_id] + ip_toks + it_toks,
            'base_con_target_ids': [bos_id] + ct_toks, 'base_incon_target_ids': [bos_id] + it_toks,
        }
        for key, seq in variants.items():
            unpadded_sequences[key].append(seq)
            global_max_len = max(global_max_len, len(seq))

        collated['con_target_start_in_con_prime'].append(len(cp_toks) + 1)
        collated['incon_target_start_in_con_prime'].append(len(cp_toks) + 1)
        collated['con_target_start_in_incon_prime'].append(len(ip_toks) + 1)
        collated['incon_target_start_in_incon_prime'].append(len(ip_toks) + 1)

        ct_hotspot_start, ct_hotspot_end = find_subsequence_indices(ct_toks, tokenized['hotspot'][i])
        it_hotspot_start, it_hotspot_end = find_subsequence_indices(it_toks, tokenized['hotspot'][i])
        collated['con_target_hotspot_start'].append(ct_hotspot_start)
        collated['con_target_hotspot_end'].append(ct_hotspot_end)
        collated['incon_target_hotspot_start'].append(it_hotspot_start)
        collated['incon_target_hotspot_end'].append(it_hotspot_end)

    for key, sequences in unpadded_sequences.items():
        collated[key] = torch.stack(
            [torch.tensor(s + [pad_id] * (global_max_len - len(s)), dtype=torch.long) for s in sequences])
    for key in list(collated.keys()):
        if isinstance(collated[key], list): collated[key] = torch.tensor(collated[key], dtype=torch.long)

    collated['metadata'] = batch
    return dict(collated)


def create_priming_dataloader(csv_path: str, tokenizer: PreTrainedTokenizer, batch_size: int, **kwargs) -> Optional[
    DataLoader]:
    processed_data = load_and_process_priming_data(csv_path=Path(csv_path))
    if not processed_data: return None
    dataset = PrimingEvaluationDataset(processed_data)
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size,
                      collate_fn=partial(collate_for_priming_eval, tokenizer=tokenizer), **kwargs)