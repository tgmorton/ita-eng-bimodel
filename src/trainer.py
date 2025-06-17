# src/trainer.py

import gc
import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Union, List, Set

import numpy as np
import torch
import torch.distributed
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import TrainingConfig


class Trainer:
    def __init__(
            self,
            config: TrainingConfig,
            model: Union[PreTrainedModel, DDP],
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
            l1_dataloader: DataLoader,
            l2_dataloader: Optional[DataLoader],
            l1_sampler: Optional[Sampler],
            l2_sampler: Optional[Sampler],
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            num_training_steps: int,
    ):
        self.config = config
        self.model = model
        # ... (rest of the assignments)
        self.tokenizer = tokenizer
        self.num_training_steps = num_training_steps

        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler(enabled=self.config.use_amp)
        self.is_main_process = (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.total_loss_since_logging = 0.0
        self.steps_since_logging = 0

        # --- FIX 1: Initialize the checkpoint schedule set ---
        self.checkpoint_schedule_set: Optional[Set[int]] = None
        if self.config.checkpoint_schedule:
            self.checkpoint_schedule_set = set(self.config.checkpoint_schedule)
            if self.is_main_process:
                self.logger.info(f"Checkpointing will occur at specified steps from the schedule ({len(self.checkpoint_schedule_set)} total).")


        if self.config.checkpoint_path:
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads training state from a checkpoint."""
        # This method remains the same as your original file
        if not self.config.checkpoint_path:
            return

        state_file = self.config.checkpoint_path / "training_state.pt"
        if not state_file.is_file():
            self.logger.warning(f"Checkpoint path specified, but {state_file} not found. Starting fresh.")
            return

        self.logger.info(f"Loading training state from: {state_file}")
        try:
            ckpt = torch.load(state_file, map_location="cpu")
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_load.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if self.scaler.is_enabled() and "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            self.current_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", 0)
            random.setstate(ckpt["python_rng_state"])
            np.random.set_state(ckpt["numpy_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])
            self.logger.info(f"Resuming training from epoch {self.current_epoch}, global step {self.global_step}.")
            del ckpt
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {state_file}: {e}", exc_info=True)
            self.current_epoch = 0
            self.global_step = 0

    def _save_checkpoint(self) -> None:
        """Saves the complete training state to a checkpoint directory."""
        # This method remains the same as your original file
        if not self.is_main_process:
            return

        ckpt_dir = self.config.output_dir / f"checkpoint-{self.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving checkpoint to {ckpt_dir}")

        unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config.model_dump(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, ckpt_dir / "training_state.pt")
        unwrapped_model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        self.logger.info(f"Checkpoint {self.global_step} saved successfully.")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs a single training step."""
        # This method remains the same as your original file
        with torch.autocast(device_type=self.device.type, enabled=self.config.use_amp):
            batch_on_device = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            outputs = self.model(**batch_on_device)
            loss = outputs.loss
        scaled_loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        return loss.item()

    def _perform_optimizer_step(self):
        """Performs one step of the optimizer and learning rate scheduler."""
        if self.config.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def train(self) -> None:
        """The main training loop with sequential data loading."""
        self.logger.info("***** Starting Sequential Training *****")

        progress_bar = tqdm(
            total=self.num_training_steps,
            initial=self.global_step,
            disable=not self.is_main_process,
            desc="Training"
        )

        # --- DEBUG-5 ---
        self.logger.info(f"[DEBUG] Trainer received l1_dataloader with length: {len(self.l1_dataloader)}")
        if self.l2_dataloader:
            self.logger.info(f"[DEBUG] Trainer received l2_dataloader with length: {len(self.l2_dataloader)}")

        for epoch in range(self.current_epoch, self.config.num_train_epochs):
            self.current_epoch = epoch
            self.model.train()

            # Set epoch for distributed samplers
            if self.l1_sampler and hasattr(self.l1_sampler, "set_epoch"):
                self.l1_sampler.set_epoch(epoch)
            if self.l2_sampler and hasattr(self.l2_sampler, "set_epoch"):
                self.l2_sampler.set_epoch(epoch)

            # --- L1 Training Loop ---
            self.logger.info(f"--- Epoch {epoch + 1}/{self.config.num_train_epochs} | Training on L1 dataset ---")

            # --- DEBUG-6 ---
            self.logger.info("[DEBUG] Entering L1 training loop...")

            for step, batch in enumerate(self.l1_dataloader):
                # --- DEBUG-7 ---
                if step == 0:
                    self.logger.info("[DEBUG] --> Successfully entered L1 loop and got the first batch!")

                loss = self._train_step(batch)
                self.total_loss_since_logging += loss
                self.steps_since_logging += 1

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._perform_optimizer_step()
                    progress_bar.update(1)

                    # --- FIX 2: Update the save condition ---
                    # This now checks for the regular interval OR if the current step is in our specific schedule
                    should_save = (self.global_step % self.config.save_steps == 0)
                    if self.checkpoint_schedule_set and self.global_step in self.checkpoint_schedule_set:
                        should_save = True

                    if should_save:
                        self._save_checkpoint()

                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

            # --- DEBUG-8 ---
            self.logger.info("[DEBUG] Exited L1 training loop.")

            # --- L2 Training Loop ---
            if self.l2_dataloader:
                self.logger.info(f"--- Epoch {epoch + 1}/{self.config.num_train_epochs} | Training on L2 dataset ---")

                # --- DEBUG-9 ---
                self.logger.info("[DEBUG] Entering L2 training loop...")

                for step, batch in enumerate(self.l2_dataloader):
                    # --- DEBUG-10 ---
                    if step == 0:
                        self.logger.info("[DEBUG] --> Successfully entered L2 loop and got the first batch!")

                    loss = self._train_step(batch)
                    self.total_loss_since_logging += loss
                    self.steps_since_logging += 1

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        self._perform_optimizer_step()
                        progress_bar.update(1)

                        # --- FIX 2: Update the save condition ---
                        # This now checks for the regular interval OR if the current step is in our specific schedule
                        should_save = (self.global_step % self.config.save_steps == 0)
                        if self.checkpoint_schedule_set and self.global_step in self.checkpoint_schedule_set:
                            should_save = True

                        if should_save:
                            self._save_checkpoint()

                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

                # --- DEBUG-11 ---
                self.logger.info("[DEBUG] Exited L2 training loop.")

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        progress_bar.close()
        self.logger.info("***** Training Finished *****")
        self._save_checkpoint()

    def save_final_model(self) -> None:
        """Saves the final model and config to a 'final_model' directory."""
        # This method remains the same as your original file
        if not self.is_main_process:
            return

        final_dir = self.config.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving final model to {final_dir}")

        unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model
        unwrapped_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        with open(final_dir / "training_config.json", "w") as f:
            f.write(self.config.model_dump_json(indent=2))
        self.logger.info("Final model and config saved successfully.")