# src/priming_evaluation/data_loader.py

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

def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    """
    Identifies the two alternating structures (e.g., 'overt', 'null')
    from the CSV column names.
    """
    structures = set()
    # Find structures from columns like 'p_overt', 't_null', etc.
    for col in columns:
        if col.startswith(('p_', 't_', 'c_', 'hotspot_')) and '_' in col:
            structure_name = col.split('_', 1)[1]
            if structure_name:
                structures.add(structure_name)

    if len(structures) == 2:
        return tuple(sorted(list(structures)))
    else:
        logger.warning(f"Could not determine exactly two structures from columns. Found: {structures}")
        return None

class PrimingEvaluationDataset(Dataset):
    """A PyTorch Dataset for priming evaluation data."""
    def __init__(self, processed_data: List[Dict[str, Any]]):
        if not isinstance(processed_data, list):
            raise TypeError(f"Expected a list for processed_data, but got {type(processed_data)}")
        self.data = processed_data
        if not self.data:
            logger.warning("PrimingEvaluationDataset initialized with no data.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

def load_and_process_priming_data(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Loads priming data from a CSV file, identifies the structures,
    and prepares it in a congruent/incongruent format for evaluation.
    This now includes processing hotspot information.
    """
    processed_data = []
    csv_filename = csv_path.name
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error loading or processing CSV {csv_filename}: {e}")
        return []

    # Detect the two alternating structures from column names
    alternation = get_structure_alternations(list(df.columns))
    if alternation is None:
        logger.error(f"Could not determine structure alternation for {csv_filename}.")
        return []

    struct1, struct2 = alternation
    logger.info(f"Detected alternation for {csv_filename}: '{struct1}' vs '{struct2}'")

    # Define required columns based on detected structures
    p_col1, p_col2 = f'p_{struct1}', f'p_{struct2}'
    t_col1, t_col2 = f't_{struct1}', f't_{struct2}'
    hotspot_col1, hotspot_col2 = f'hotspot_{struct1}', f'hotspot_{struct2}'
    c_col1, c_col2 = f'c_{struct1}', f'c_{struct2}'

    required_cols = [p_col1, p_col2, t_col1, t_col2, hotspot_col1, hotspot_col2, c_col1, c_col2]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"CSV {csv_filename} is missing required columns: {missing}")
        return []

    logger.info(f"Processing {len(df)} rows from {csv_filename}...")
    for index, row in df.iterrows():
        try:
            # Create two items per row, one for each structure as the target
            item1 = {
                'congruent_prime': str(row[c_col1]) + " " + str(row[p_col1]),
                'incongruent_prime': str(row[c_col1]) + " " + str(row[p_col2]),
                'congruent_target': str(row[t_col1]),
                'incongruent_target': str(row[t_col2]),
                'congruent_hotspot': str(row[hotspot_col1]),
                'incongruent_hotspot': str(row[hotspot_col2]),
                'target_structure': struct1,
                'source_csv': csv_filename,
                'csv_row': index,
            }
            item2 = {
                'congruent_prime': str(row[c_col2]) + " " + str(row[p_col2]),
                'incongruent_prime': str(row[c_col2]) + " " + str(row[p_col1]),
                'congruent_target': str(row[t_col2]),
                'incongruent_target': str(row[t_col1]),
                'congruent_hotspot': str(row[hotspot_col2]),
                'incongruent_hotspot': str(row[hotspot_col1]),
                'target_structure': struct2,
                'source_csv': csv_filename,
                'csv_row': index,
            }
            processed_data.extend([item1, item2])
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping row {index} in {csv_filename} due to error: {e}")
            continue

    logger.info(f"Finished processing {csv_filename}. Created {len(processed_data)} valid items.")
    return processed_data


def find_subsequence_indices(sequence, subsequence):
    """Finds the start and end indices of a subsequence within a sequence."""
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i+len(subsequence)] == subsequence:
            return i, i + len(subsequence)
    return -1, -1

def collate_for_priming_eval(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """
    Collates data for priming evaluation. It assembles 6 sequence variations,
    pads them to a unified length, and calculates start/end indices for both
    the full target and the specific hotspot within the target.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    collated = defaultdict(list)
    bos_id, pad_id = tokenizer.bos_token_id, tokenizer.pad_token_id
    if bos_id is None or pad_id is None:
        logger.error("Tokenizer requires both a BOS and a PAD token for this evaluation.")
        return {}

    tokenized_parts = defaultdict(list)
    for item in batch:
        for key in ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target', 'congruent_hotspot', 'incongruent_hotspot']:
            tokenized_parts[key].append(tokenizer(item[key], add_special_tokens=False)['input_ids'])

    max_len = 0
    sequences_to_pad = defaultdict(list)

    for i in range(len(batch)):
        cp_toks, ip_toks = tokenized_parts['congruent_prime'][i], tokenized_parts['incongruent_prime'][i]
        ct_toks, it_toks = tokenized_parts['congruent_target'][i], tokenized_parts['incongruent_target'][i]
        ct_hotspot_toks, it_hotspot_toks = tokenized_parts['congruent_hotspot'][i], tokenized_parts['incongruent_hotspot'][i]

        # Assemble the 6 sequence variations
        variants = {
            'con_prime_con_target_ids': [bos_id] + cp_toks + ct_toks,
            'con_prime_incon_target_ids': [bos_id] + cp_toks + it_toks,
            'incon_prime_con_target_ids': [bos_id] + ip_toks + ct_toks,
            'incon_prime_incon_target_ids': [bos_id] + ip_toks + it_toks,
            'base_con_target_ids': [bos_id] + ct_toks,
            'base_incon_target_ids': [bos_id] + it_toks,
        }

        for key, seq in variants.items():
            sequences_to_pad[key].append(torch.tensor(seq))
            max_len = max(max_len, len(seq))

        # Store target start indices (after BOS token)
        collated['con_target_start_in_con_prime'].append(len(cp_toks) + 1)
        collated['incon_target_start_in_con_prime'].append(len(cp_toks) + 1)
        collated['con_target_start_in_incon_prime'].append(len(ip_toks) + 1)
        collated['incon_target_start_in_incon_prime'].append(len(ip_toks) + 1)

        # Find and store hotspot indices within their respective targets
        ct_hotspot_start, ct_hotspot_end = find_subsequence_indices(ct_toks, ct_hotspot_toks)
        it_hotspot_start, it_hotspot_end = find_subsequence_indices(it_toks, it_hotspot_toks)
        collated['con_target_hotspot_start'].append(ct_hotspot_start)
        collated['con_target_hotspot_end'].append(ct_hotspot_end)
        collated['incon_target_hotspot_start'].append(it_hotspot_start)
        collated['incon_target_hotspot_end'].append(it_hotspot_end)

    # Pad all sequences to the same max length and stack them
    for key, seq_list in sequences_to_pad.items():
        collated[key] = pad_sequence(seq_list, batch_first=True, padding_value=pad_id)

    # Convert index lists to tensors
    for key in list(collated.keys()):
        if 'start' in key or 'end' in key:
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