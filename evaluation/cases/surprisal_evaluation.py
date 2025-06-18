# evaluation/cases/surprisal_evaluation.py (DEBUG Version)

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
    This version includes robust error handling and detailed logging for debugging.
    """

    def _analyze_sentence(self, context: str, target: str, hotspot: str) -> Dict[str, Any]:
        """Analyzes a single sentence (context + target) for surprisal."""
        # This helper function now returns a more detailed dictionary for better debugging
        analysis_result = {
            "full_text": "N/A", "full_tokens": [], "full_surprisals": [],
            "hotspot_analysis": {}, "error": None
        }
        try:
            clean_context = str(context).strip()
            clean_target = str(target).strip()
            analysis_result["full_text"] = f"{clean_context} {clean_target}"

            tokens, surprisals, offset_mapping = self.model_wrapper.get_surprisals(analysis_result["full_text"])
            analysis_result["full_tokens"] = tokens
            analysis_result["full_surprisals"] = surprisals

            stripped_hotspot = str(hotspot).strip()
            hotspot_char_start_in_target = clean_target.find(stripped_hotspot)

            if hotspot_char_start_in_target != -1:
                context_len = len(clean_context) + 1
                hotspot_char_start_in_full = context_len + hotspot_char_start_in_target
                hotspot_char_end_in_full = hotspot_char_start_in_full + len(stripped_hotspot)

                hotspot_indices = [i for i, (start, end) in enumerate(offset_mapping) if
                                   start < hotspot_char_end_in_full and end > hotspot_char_start_in_full]

                if hotspot_indices:
                    hotspot_surprisals = [surprisals[i] for i in hotspot_indices]
                    analysis_result["hotspot_analysis"] = {
                        "avg_surprisal": np.mean(hotspot_surprisals).item(),
                        "sum_surprisal": np.sum(hotspot_surprisals).item(),
                        "num_tokens": len(hotspot_surprisals),
                        "tokens": [tokens[i] for i in hotspot_indices]
                    }
            else:
                logger.warning(f"Hotspot '{stripped_hotspot}' not found in target '{clean_target}'.")

        except Exception as e:
            logger.error(f"Error during sentence analysis for text '{analysis_result['full_text']}': {e}")
            analysis_result["error"] = str(e)

        return analysis_result

    def run(self, data: pd.DataFrame, source_filename: str = "unknown") -> List[Dict]:
        if data.empty:
            logger.warning(f"Skipping surprisal run for {source_filename} because no data was loaded.")
            return []

        results = []
        for index, row in tqdm(data.iterrows(), total=len(data), desc=f"Surprisal for {source_filename}"):
            try:
                null_analysis = self._analyze_sentence(row["context"], row["null_sentence"], row["hotspot"])
                overt_analysis = self._analyze_sentence(row["context"], row["overt_sentence"], row["hotspot"])

                diff_score = None
                if null_analysis.get("hotspot_analysis", {}).get("avg_surprisal") is not None and \
                        overt_analysis.get("hotspot_analysis", {}).get("avg_surprisal") is not None:
                    diff_score = (
                            null_analysis["hotspot_analysis"]["avg_surprisal"] -
                            overt_analysis["hotspot_analysis"]["avg_surprisal"]
                    )

                results.append({
                    "item_id": row.get("item_id", index),
                    "source_file": source_filename,
                    "context": row["context"],
                    "hotspot_text": row["hotspot"],
                    "null_sentence_full": row["null_sentence"],
                    "overt_sentence_full": row["overt_sentence"],
                    "null_sentence_analysis": null_analysis,
                    "overt_sentence_analysis": overt_analysis,
                    "hotspot_difference_score": diff_score,
                })
            except Exception as e:
                logger.error(
                    f"CRITICAL: Failed to process item_id '{row.get('item_id', index)}' from {source_filename}. Error: {e}")
                # Append a record of the failure
                results.append({"item_id": row.get("item_id', index"), "source_file": source_filename, "error": str(e)})

        logger.info(f"Finished processing {len(results)} items for {source_filename}.")
        return results