# src/data.py

import logging
from typing import Optional, Tuple

from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .config import TrainingConfig


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
            logger.info(f"Successfully loaded dataset with {len(dataset):,} samples.")
        except FileNotFoundError:
            logger.error(f"Dataset not found at path: {dataset_path}")
            raise

        block_size = 1024

        def chunk(batch):
            new_rows = []
            for ids in batch["input_ids"]:
                pieces = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
                new_rows.extend(pieces)
            return {"input_ids": new_rows}

        orig_cols = dataset.column_names

        dataset = (
            dataset.map(
                chunk,
                batched=True,
                remove_columns=orig_cols,
                batch_size=1000,
                desc=f"Chunking {dataset_path}",
            )
            .filter(
                lambda x: len(x["input_ids"]) <= block_size,
                desc=f"Filtering {dataset_path}",
            )
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        sampler: Optional[Sampler] = None
        if is_distributed:
            sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)
        else:
            sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.per_device_train_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=data_collator,
            persistent_workers=(True if config.num_workers > 0 else False),
        )
        return dataloader, sampler

    if not config.l1_dataset_path:
        raise ValueError("At least `l1_dataset_path` must be specified in the configuration.")

    l1_dataloader, l1_sampler = _create_single_dataloader(config.l1_dataset_path)
    l2_dataloader, l2_sampler = _create_single_dataloader(config.l2_dataset_path)

    return l1_dataloader, l2_dataloader, l1_sampler, l2_sampler