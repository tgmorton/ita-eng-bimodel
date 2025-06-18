# evaluation/data_loader_priming.py (Corrected and Complete)

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
    if sub_len == 0: return -1, -1
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
        df = pd.read_csv(csv_path).fillna('')
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
            prime_context, target_context = str(row[c_p_col]), str(row[c_t_col])
            hotspot = str(row[hotspot_t_col])

            prime1, prime2 = str(row[p_col1]), str(row[p_col2])
            target1, target2 = str(row[t_col1]), str(row[t_col2])

            # Pair 1: struct1 is congruent
            processed_data.append({
                'congruent_prime': f"{prime_context} {prime1}".strip(),
                'incongruent_prime': f"{prime_context} {prime2}".strip(),
                'congruent_target': f"{target_context} {target1}".strip(),
                'incongruent_target': f"{target_context} {target2}".strip(),
                'hotspot': hotspot,
                'target_structure': struct1
            })
            # Pair 2: struct2 is congruent
            processed_data.append({
                'congruent_prime': f"{prime_context} {prime2}".strip(),
                'incongruent_prime': f"{prime_context} {prime1}".strip(),
                'congruent_target': f"{target_context} {target2}".strip(),
                'incongruent_target': f"{target_context} {target1}".strip(),
                'hotspot': hotspot,
                'target_structure': struct2
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping row {index} in {csv_filename} due to error: {e}")

    return processed_data


def collate_for_priming_eval(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Collates data by tokenizing, assembling 6 sequence variations, finding a GLOBAL max
    length for the batch, padding all sequences to it, and calculating indices.
    """
    batch = [item for item in batch if item]
    if not batch: return {}

    collated = defaultdict(list)
    bos_id, pad_id = tokenizer.bos_token_id, tokenizer.pad_token_id
    if bos_id is None or pad_id is None:
        logger.error("Tokenizer requires BOS and PAD tokens.");
        return {}

    tokenized, unpadded_sequences = defaultdict(list), defaultdict(list)
    global_max_len = 0

    # Tokenize all parts first
    for item in batch:
        for key in ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target', 'hotspot']:
            tokenized[key].append(tokenizer(item[key], add_special_tokens=False)['input_ids'])

    # Assemble sequences and find the single max length for the entire batch
    for i in range(len(batch)):
        cp_toks, ip_toks = tokenized['congruent_prime'][i], tokenized['incongruent_prime'][i]
        ct_toks, it_toks = tokenized['congruent_target'][i], tokenized['incongruent_target'][i]

        variants = {
            'con_prime_con_target_ids': [bos_id] + cp_toks + ct_toks,
            'con_prime_incon_target_ids': [bos_id] + cp_toks + it_toks,
            'incon_prime_con_target_ids': [bos_id] + ip_toks + ct_toks,
            'incon_prime_incon_target_ids': [bos_id] + ip_toks + it_toks,
            'base_con_target_ids': [bos_id] + ct_toks,
            'base_incon_target_ids': [bos_id] + it_toks,
        }
        for key, seq in variants.items():
            unpadded_sequences[key].append(seq)
            global_max_len = max(global_max_len, len(seq))

        # Calculate and store indices
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

    # Pad all sequences to the global max length and create final tensors
    for key, sequences in unpadded_sequences.items():
        padded_tensors = []
        for seq in sequences:
            padded_seq = seq + ([pad_id] * (global_max_len - len(seq)))
            padded_tensors.append(torch.tensor(padded_seq, dtype=torch.long))
        collated[key] = torch.stack(padded_tensors)

    # Convert index lists to tensors
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