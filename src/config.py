# src/config.py

from enum import Enum
from pathlib import Path
from typing import Literal, Optional, List

from pydantic import (BaseModel, DirectoryPath, FilePath, PositiveInt,
                      field_validator)
from pydantic_core import PydanticCustomError


# Define the choices for the learning rate scheduler using an Enum
class LRSchedulerType(str, Enum):
    linear = "linear"
    cosine = "cosine"
    constant = "constant"
    constant_with_warmup = "constant_with_warmup"


class TrainingConfig(BaseModel):
    """
    Configuration for the training script, validated by Pydantic.
    """
    # === Essential Paths ===
    l1_dataset_path: DirectoryPath
    l2_dataset_path: Optional[DirectoryPath] = None
    output_dir: Path
    checkpoint_path: Optional[DirectoryPath] = None

    # === Model Configuration ===
    tokenizer_path: str
    model_arch_type: str = 'gpt2'
    train_from_scratch: bool = True
    model_size_tag: Optional[str] = None
    architectures_path: FilePath = Path("configs/model_architectures.yaml")

    # === Training Hyperparameters ===
    num_train_epochs: PositiveInt = 3
    max_steps: int = -1
    per_device_train_batch_size: PositiveInt = 8
    gradient_accumulation_steps: PositiveInt = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: LRSchedulerType = "linear"
    num_warmup_steps: int = 0

    # === Hardware & Precision ===
    use_amp: bool = False
    num_workers: int = 4

    # === Control & Reproducibility ===
    seed: int = 42

    # === Logging & Saving ===
    logging_steps: PositiveInt = 100
    save_steps: PositiveInt = 500

    checkpoint_schedule: Optional[List[int]] = None

    @field_validator("model_size_tag")
    @classmethod
    def check_scratch_params(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get("train_from_scratch") and v is None:
            raise PydanticCustomError(
                "missing_model_size_tag",
                "`model_size_tag` is required when `train_from_scratch` is True.",
            )
        return v

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v