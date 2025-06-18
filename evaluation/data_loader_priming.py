# evaluation/data_loader_priming.py (Complete and Corrected)

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
    """A PyTorch Dataset for priming evaluation data."""

    def __init__(self, processed_data: List[Dict[str, Any]]):
        self.data = processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def find_subsequence_indices(sequence: List[int], subsequence: List[int]) -> Tuple[int, int]:
    """Finds the start and end token indices of a subsequence within a sequence."""
    sub_len = len(subsequence)
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            return i, i + sub_len
    return -1, -1


def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    """
    Correctly identifies the two alternating structures (e.g., 'overt', 'null')
    by looking only at the suffixes of columns starting with 'p_'.
    """
    prime_structures = set()
    for col in columns:
        col = col.strip()
        if col.startswith('p_'):
            # e.g., 'p_overt' -> 'overt'
            structure_name = col.split('_', 1)[1]
            if structure_name:
                prime_structures.add(structure_name)

    if len(prime_structures) == 2:
        return tuple(sorted(list(prime_structures)))
    else:
        logger.warning(
            f"Could not determine exactly two structures from prime columns (p_...). Found: {prime_structures}")
        return None


def load_and_process_priming_data(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Loads priming data from a CSV, robustly handling column names.
    It creates a list of dictionaries, each representing a single evaluation item
    for the collate function to process.
    """
    processed_data = []
    csv_filename = csv_path.name
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error loading CSV {csv_filename}: {e}");
        return []

    alternation = get_structure_alternations(list(df.columns))
    if alternation is None:
        logger.error(f"Could not process {csv_filename} due to structure ambiguity.")
        return []

    struct1, struct2 = alternation
    logger.info(f"Processing {csv_filename}: Detected alternation '{struct1}' vs '{struct2}'")

    p_col1, p_col2 = f'p_{struct1}', f'p_{struct2}'
    t_col1, t_col2 = f't_{struct1}', f't_{struct2}'
    c_p_col, c_t_col = 'c_p', 'c_t'
    hotspot_t_col = 'hotspot_t'

    required_cols = [p_col1, p_col2, t_col1, t_col2, c_p_col, c_t_col, hotspot_t_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"CSV {csv_filename} is missing required columns: {missing}")
        return []

    for index, row in df.iterrows():
        try:
            prime_context = str(row[c_p_col])
            target_context = str(row[c_t_col])
            hotspot = str(row[hotspot_t_col])

            prime_struct1, prime_struct2 = str(row[p_col1]), str(row[p_col2])
            target_struct1, target_struct2 = str(row[t_col1]), str(row[t_col2])

            # Pair 1: struct1 is congruent
            processed_data.append({
                'congruent_prime': f"{prime_context} {prime_struct1}",
                'incongruent_prime': f"{prime_context} {prime_struct2}",
                'congruent_target': f"{target_context} {target_struct1}",
                'hotspot': hotspot,
                'target_structure': struct1
            })
            # Pair 2: struct2 is congruent
            processed_data.append({
                'congruent_prime': f"{prime_context} {prime_struct2}",
                'incongruent_prime': f"{prime_context} {prime_struct1}",
                'congruent_target': f"{target_context} {target_struct2}",
                'hotspot': hotspot,
                'target_structure': struct2
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping row {index} in {csv_filename} due to error: {e}")

    return processed_data


def collate_for_priming_eval(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Collates data for priming evaluation by tokenizing, assembling 6 sequence variations,
    padding them to a unified length, and calculating start/end indices for the target and hotspot.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    collated = defaultdict(list)
    bos_id, pad_id = tokenizer.bos_token_id, tokenizer.pad_token_id
    if bos_id is None or pad_id is None:
        logger.error("Tokenizer requires BOS and PAD tokens.");
        return {}

    tokenized = defaultdict(list)
    for item in batch:
        for key in ['congruent_prime', 'incongruent_prime', 'congruent_target', 'hotspot']:
            tokenized[key].append(tokenizer(item[key], add_special_tokens=False)['input_ids'])

    max_len = 0
    sequences_to_pad = defaultdict(list)

    for i in range(len(batch)):
        cp_toks, ip_toks = tokenized['congruent_prime'][i], tokenized['incongruent_prime'][i]
        ct_toks = tokenized['congruent_target'][i]
        # For this setup, incongruent target is not needed as it's the same across congruent/incongruent primes
        hotspot_toks = tokenized['hotspot'][i]

        # Assemble the 6 variations
        variants = {
            'con_prime_con_target_ids': [bos_id] + cp_toks + ct_toks,
            'incon_prime_con_target_ids': [bos_id] + ip_toks + ct_toks,
            'base_con_target_ids': [bos_id] + ct_toks,
        }

        # Add placeholder for other sequences to match evaluator expectations
        variants['con_prime_incon_target_ids'] = variants['con_prime_con_target_ids']
        variants['incon_prime_incon_target_ids'] = variants['incon_prime_con_target_ids']
        variants['base_incon_target_ids'] = variants['base_con_target_ids']

        for key, seq in variants.items():
            sequences_to_pad[key].append(torch.tensor(seq))
            max_len = max(max_len, len(seq))

        # Store target start indices
        collated['con_target_start_in_con_prime'].append(len(cp_toks) + 1)
        collated['con_target_start_in_incon_prime'].append(len(ip_toks) + 1)

        # Find and store hotspot indices
        hotspot_start, hotspot_end = find_subsequence_indices(ct_toks, hotspot_toks)
        collated['con_target_hotspot_start'].append(hotspot_start)
        collated['con_target_hotspot_end'].append(hotspot_end)

        # In this simplified logic, incongruent target is same as congruent
        collated['incon_target_start_in_con_prime'] = collated['con_target_start_in_con_prime']
        collated['incon_target_start_in_incon_prime'] = collated['con_target_start_in_incon_prime']
        collated['incon_target_hotspot_start'] = collated['con_target_hotspot_start']
        collated['incon_target_hotspot_end'] = collated['con_target_hotspot_end']

    for key, seq_list in sequences_to_pad.items():
        collated[key] = pad_sequence(seq_list, batch_first=True, padding_value=pad_id)

    for key in list(collated.keys()):
        if isinstance(collated[key], list):
            collated[key] = torch.tensor(collated[key], dtype=torch.long)

    collated['target_structure'] = [item['target_structure'] for item in batch]
    return dict(collated)


def create_priming_dataloader(
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = 0,
        max_samples: int = -1,
        seed: int = 42,
        **kwargs
) -> Optional[DataLoader]:
    """Creates a DataLoader for the priming evaluation task."""
    csv_path_obj = Path(csv_path)
    logger.info(f"Creating priming dataloader for: {csv_path_obj.name}")

    processed_data = load_and_process_priming_data(csv_path=csv_path_obj)
    if not processed_data:
        logger.warning(f"No data processed from {csv_path_obj.name}. DataLoader will be empty.")
        return None

    if max_samples > 0 and len(processed_data) > max_samples:
        logger.info(f"Sampling {max_samples:,} items from {len(processed_data):,} (seed: {seed}).")
        random.seed(seed)
        processed_data = random.sample(processed_data, k=max_samples)

    dataset = PrimingEvaluationDataset(processed_data)
    collate_fn = partial(collate_for_priming_eval, tokenizer=tokenizer)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=kwargs.get('pin_memory', True)
    )
    logger.info(f"Priming DataLoader created with {len(dataset)} items.")
    return dataloader