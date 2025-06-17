# src/train.py

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import typer
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import PreTrainedModel, get_scheduler

# Import our refactored components
from .config import LRSchedulerType, TrainingConfig
from .data import create_dataloaders
from .model import create_model_and_tokenizer
from .trainer import Trainer
from .utils import get_device, set_seed, setup_distributed, setup_logging

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def create_optimizer_and_scheduler(
        model: PreTrainedModel, config: TrainingConfig, num_training_steps: int
) -> Tuple[Optimizer, _LRScheduler]:
    logger = logging.getLogger(__name__)
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if
                    not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if
                    any(nd in n.lower() for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    logger.info(f"Optimizer created: AdamW with learning rate {config.learning_rate}")
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type.value,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"LR Scheduler created: {config.lr_scheduler_type.value}")
    return optimizer, lr_scheduler


@app.command()
def main(
        config_file: Optional[Path] = typer.Option(None, help="Path to a YAML configuration file."),
        l1_dataset_path: Optional[Path] = typer.Option(None, help="Path to the L1 training Arrow dataset."),
        l2_dataset_path: Optional[Path] = typer.Option(None, help="Path to the L2 training Arrow dataset."),
        output_dir: Optional[Path] = typer.Option(None, help="Directory for checkpoints, logs, final model."),
        tokenizer_path: Optional[str] = typer.Option(None, help="Path to the trained tokenizer directory."),
        model_arch_type: Optional[str] = typer.Option(None,
                                                      help="Base model architecture type (e.g., 'gpt2', 'llama')."),
        checkpoint_path: Optional[Path] = typer.Option(None, help="Path to checkpoint to RESUME training from."),
        train_from_scratch: Optional[bool] = typer.Option(None, help="Train a new model from scratch."),
        model_size_tag: Optional[str] = typer.Option(None, help="Tag for model architecture from YAML file."),
        num_train_epochs: Optional[int] = typer.Option(None, help="Total training epochs."),
        max_steps: Optional[int] = typer.Option(None, help="Maximum number of optimizer steps to train for."),
        per_device_train_batch_size: Optional[int] = typer.Option(None, help="Train batch size per device."),
        gradient_accumulation_steps: Optional[int] = typer.Option(None, help="Steps for gradient accumulation."),
        learning_rate: Optional[float] = typer.Option(None, help="Peak learning rate."),
        weight_decay: Optional[float] = typer.Option(None, help="Weight decay coefficient."),
        max_grad_norm: Optional[float] = typer.Option(None, help="Max gradient norm for clipping."),
        lr_scheduler_type: Optional[LRSchedulerType] = typer.Option(None, help="LR scheduler type."),
        num_warmup_steps: Optional[int] = typer.Option(None, help="LR warmup steps."),
        use_amp: Optional[bool] = typer.Option(None, help="Enable AMP training."),
        num_workers: Optional[int] = typer.Option(None, help="DataLoader workers."),
        seed: Optional[int] = typer.Option(None, help="Random seed."),
        logging_steps: Optional[int] = typer.Option(None, help="Log train metrics every X steps."),
        save_steps: Optional[int] = typer.Option(None, help="Save checkpoint every X steps."),
):
    """
    Main entry point for the training script.
    Orchestrates the entire training process by initializing components
    and running the Trainer.
    """
    if config_file:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}

    cli_args = {k: v for k, v in locals().items() if v is not None and k != 'config_data'}
    config_data.update(cli_args)
    config = TrainingConfig(**config_data)

    is_distributed, rank, _, _ = setup_distributed()
    setup_logging(rank=rank)
    set_seed(config.seed + rank)
    device = get_device()
    logger = logging.getLogger(__name__)

    if rank == 0:
        logger.info("***** Starting Training *****")
        logger.info(f"  Final Config: {config.model_dump_json(indent=2)}")

    model, tokenizer = create_model_and_tokenizer(config)
    model.to(device)

    l1_dataloader, l2_dataloader, l1_sampler, l2_sampler = create_dataloaders(
        config=config, tokenizer=tokenizer, is_distributed=is_distributed
    )

    if config.max_steps and config.max_steps > 0:
        num_training_steps = config.max_steps
    else:
        len_l1 = len(l1_dataloader)
        len_l2 = len(l2_dataloader) if l2_dataloader else 0
        num_optimizer_steps_per_epoch = (len_l1 + len_l2) // config.gradient_accumulation_steps
        num_training_steps = config.num_train_epochs * num_optimizer_steps_per_epoch

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model, config=config, num_training_steps=num_training_steps
    )

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        l1_dataloader=l1_dataloader,
        l2_dataloader=l2_dataloader,
        l1_sampler=l1_sampler,
        l2_sampler=l2_sampler,
        device=device,
        tokenizer=tokenizer,
        num_training_steps=num_training_steps
    )

    try:
        trainer.train()
        trainer.save_final_model()
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during training: {e}", exc_info=True)
    finally:
        if is_distributed:
            torch.distributed.destroy_process_group()
        logger.info(f"Training script finished on rank {rank}.")


if __name__ == "__main__":
    app()