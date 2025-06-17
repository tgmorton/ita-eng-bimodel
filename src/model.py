# src/model.py

# === Imports ===
import logging
import yaml
from pathlib import Path
from typing import Any, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,  # Changed from GPT2LMHeadModel
    PreTrainedTokenizer,
    PretrainedConfig,
)

from .config import TrainingConfig

# --- Tokenizer Creation (no change) ---
def create_tokenizer(model_name_or_path: str) -> PreTrainedTokenizer:
    # This function is already flexible enough thanks to AutoTokenizer.
    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer `pad_token` was None. Set to `eos_token`: '{tokenizer.eos_token}'")
        else:
            pad_token_str = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token_str})
            logger.warning(
                f"Tokenizer had no `pad_token` or `eos_token`. Added a new `pad_token`: '{pad_token_str}'"
            )
    return tokenizer


# --- Model Creation (Updated with flexible logic) ---

def _create_config_from_scratch(config: TrainingConfig) -> PretrainedConfig:
    """
    Creates a model config for training from scratch.
    It loads a base architecture and overrides it with parameters from a YAML file.
    """
    logger = logging.getLogger(__name__)

    # Load the base config for the desired architecture type (e.g., 'gpt2', 'llama')
    # `model_name_or_path` now specifies the *type* of model to build.
    base_config = AutoConfig.from_pretrained(config.model_arch_type)
    logger.info(f"Loaded base config structure from: '{config.model_arch_type}'")

    # Load all available architecture definitions
    with open(config.architectures_path, 'r') as f:
        architectures = yaml.safe_load(f)

    # Get the parameters for the selected model size tag
    arch_params = architectures.get(config.model_size_tag)
    if not arch_params:
        raise ValueError(
            f"Model tag '{config.model_size_tag}' not found in {config.architectures_path}."
        )

    logger.info(f"Applying config modifications for tag: '{config.model_size_tag}'")

    # Override the base config with our custom parameters
    for key, value in arch_params.items():
        setattr(base_config, key, value)

    logger.info(f"Final config params: {arch_params}")
    return base_config


def create_model_and_tokenizer(
        config: TrainingConfig,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Creates and returns the model and tokenizer based on the training mode.

    Args:
        config: The training configuration object.

    Returns:
        A tuple containing the model and the tokenizer.
    """
    logger = logging.getLogger(__name__)

    # Load tokenizer
    tokenizer = create_tokenizer(config.tokenizer_path)

    if config.train_from_scratch:
        logger.info("Mode: Training new model from scratch.")
        model_config = _create_config_from_scratch(config)
        # Set vocab size from the tokenizer, as it might have been expanded
        model_config.vocab_size = len(tokenizer)

        # Use AutoModelForCausalLM.from_config to create a new model instance
        model = AutoModelForCausalLM.from_config(model_config)
        logger.info("Model architecture initialized with random weights.")

    else:
        logger.info(f"Mode: Fine-tuning pre-trained model from '{config.model_name_or_path}'.")
        # Use AutoModelForCausalLM.from_pretrained to load an existing model
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

        # IMPORTANT: Resize model embeddings if the tokenizer was expanded
        # This is crucial for models where we added a pad token.
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model token embeddings to: {len(tokenizer)}")

    return model, tokenizer