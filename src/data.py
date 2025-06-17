# src/data.py

import logging
from pathlib import Path
from typing import Optional, Tuple

from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .config import TrainingConfig


def _chunk_examples(batch, block_size: int):
    """
    Concatenates all examples in a batch and then chunks them into blocks
    of a specified size. This is a more robust method for use with map.
    """
    # 1. Concatenate all texts from the batch
    concatenated_examples = {k: sum(batch[k], []) for k in batch.keys()}
    total_length = len(concatenated_examples[list(batch.keys())[0]])

    # 2. Drop the remainder that is smaller than the block_size
    if total_length < block_size:
        # Return an empty dict if the concatenated batch is too small
        # This will be filtered out by the subsequent .filter() call
        return {k: [] for k in batch.keys()}

    total_length = (total_length // block_size) * block_size

    # 3. Split into chunks of block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def create_dataloaders(
        config: TrainingConfig,
        tokenizer: PreTrainedTokenizer,
        is_distributed: bool,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[Sampler], Optional[Sampler]]:
    """
    Loads one or two pre-processed datasets from disk and creates DataLoaders.
    Returns dataloaders and samplers for L1 and L2.
    """
    logger = logging.getLogger(__name__)

    def _create_single_dataloader(
            dataset_path: Optional[Path],
    ) -> Tuple[Optional[DataLoader], Optional[Sampler]]:
        """Helper function to create a DataLoader for a single dataset."""
        if not dataset_path:
            return None, None

        logger.info(f"Loading data from: {dataset_path}")
        try:
            dataset = load_from_disk(str(dataset_path))
            # --- DEBUG-1 ---
            logger.info(f"[DEBUG] Initial dataset length for {dataset_path.name}: {len(dataset):,}")
        except FileNotFoundError:
            logger.error(f"Dataset not found at path: {dataset_path}")
            raise

        block_size = 1024
        orig_cols = dataset.column_names

        mapped_dataset = dataset.map(
            _chunk_examples,
            batched=True,
            fn_kwargs={'block_size': block_size},
            remove_columns=orig_cols,
            batch_size=1000,
            desc=f"Chunking {dataset_path.name}",
        )
        # --- DEBUG-2 ---
        logger.info(f"[DEBUG] Dataset length after .map() for {dataset_path.name}: {len(mapped_dataset):,}")

        filtered_dataset = mapped_dataset.filter(
            lambda x: len(x["input_ids"]) > 0,
            desc=f"Filtering {dataset_path.name}",
        )
        # --- DEBUG-3 ---
        logger.info(f"[DEBUG] Dataset length after .filter() for {dataset_path.name}: {len(filtered_dataset):,}")

        logger.info(f"Finished processing {dataset_path.name}, resulting in {len(filtered_dataset):,} samples.")

        if len(filtered_dataset) == 0:
            logger.warning(f"Dataset at {dataset_path} is empty after processing. The dataloader will be empty.")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        sampler: Optional[Sampler] = None
        if is_distributed:
            sampler = DistributedSampler(filtered_dataset, shuffle=True, seed=config.seed)
        else:
            sampler = RandomSampler(filtered_dataset)

        dataloader = DataLoader(
            filtered_dataset,
            sampler=sampler,
            batch_size=config.per_device_train_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=data_collator,
            persistent_workers=(True if config.num_workers > 0 else False),
        )

        # --- DEBUG-4 ---
        logger.info(f"[DEBUG] Final DataLoader length for {dataset_path.name}: {len(dataloader):,}")

        return dataloader, sampler

    if not config.l1_dataset_path:
        raise ValueError("At least `l1_dataset_path` must be specified in the configuration.")

    l1_dataloader, l1_sampler = _create_single_dataloader(config.l1_dataset_path)
    l2_dataloader, l2_sampler = _create_single_dataloader(config.l2_dataset_path)

    return l1_dataloader, l2_dataloader, l1_sampler, l2_sampler