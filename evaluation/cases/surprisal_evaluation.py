# evaluation/cases/surprisal_evaluation.py (Corrected and Enhanced)

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import logging

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class SurprisalEvaluation(EvaluationCase):
    """
    A concrete evaluation case for calculating surprisal on null vs. overt pronouns.
    This version provides detailed, item-level output.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _analyze_sentence(self, context: str, target: str, hotspot: str) -> Dict[str, Any]:
        """Analyzes a single sentence (context + target) for surprisal."""
        # Ensure inputs are clean strings
        clean_context = str(context).strip()
        clean_target = str(target).strip()
        full_text = f"{clean_context} {clean_target}"

        try:
            tokens, surprisals, offset_mapping = self.model_wrapper.get_surprisals(full_text)
        except Exception as e:
            logger.error(f"Failed to get surprisals for text: '{full_text}'. Error: {e}")
            return {}

        hotspot_analysis = {}
        stripped_hotspot = str(hotspot).strip()

        try:
            # Find the character position of the hotspot within the target sentence
            hotspot_char_start_in_target = clean_target.find(stripped_hotspot)
            if hotspot_char_start_in_target != -1:
                context_len = len(clean_context) + 1  # +1 for the space
                hotspot_char_start_in_full = context_len + hotspot_char_start_in_target
                hotspot_char_end_in_full = hotspot_char_start_in_full + len(stripped_hotspot)

                # Find all tokens that overlap with the hotspot's character span
                hotspot_indices = [
                    i for i, (start, end) in enumerate(offset_mapping)
                    if start < hotspot_char_end_in_full and end > hotspot_char_start_in_full
                ]

                if hotspot_indices:
                    hotspot_surprisals = [surprisals[i] for i in hotspot_indices]
                    hotspot_analysis = {
                        "avg_surprisal": np.mean(hotspot_surprisals).item() if hotspot_surprisals else None,
                        "sum_surprisal": np.sum(hotspot_surprisals).item() if hotspot_surprisals else None,
                        "num_tokens": len(hotspot_surprisals),
                        "tokens": [tokens[i] for i in hotspot_indices]
                    }
        except Exception as e:
            logger.error(f"Failed during hotspot analysis for hotspot '{stripped_hotspot}'. Error: {e}")

        return {
            "full_text": full_text,
            "full_tokens": tokens,
            "full_surprisals": surprisals,
            "hotspot_analysis": hotspot_analysis
        }

    def run(self, data: pd.DataFrame, source_filename: str = "unknown") -> List[Dict]:
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Surprisal for {source_filename}"):
            # Analyze both the null and overt sentence variations
            null_analysis = self._analyze_sentence(row["context"], row["null_sentence"], row["hotspot"])
            overt_analysis = self._analyze_sentence(row["context"], row["overt_sentence"], row["hotspot"])

            # Calculate difference score if possible
            diff_score = None
            if null_analysis.get("hotspot_analysis", {}).get("avg_surprisal") is not None and \
                    overt_analysis.get("hotspot_analysis", {}).get("avg_surprisal") is not None:
                diff_score = (
                        null_analysis["hotspot_analysis"]["avg_surprisal"] -
                        overt_analysis["hotspot_analysis"]["avg_surprisal"]
                )

            # Append detailed item-level results
            results.append({
                "item_id": row.get("item_id", "N/A"),
                "source_file": source_filename,
                "context": row["context"],
                "hotspot_text": row["hotspot"],
                "null_sentence_full": row["null_sentence"],
                "overt_sentence_full": row["overt_sentence"],
                "null_sentence_analysis": null_analysis,
                "overt_sentence_analysis": overt_analysis,
                "hotspot_difference_score": diff_score,
            })
        return results