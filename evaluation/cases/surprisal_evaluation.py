# evaluation/cases/surprisal_evaluation.py (Final Version)

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
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _analyze_sentence(self, context: str, target: str, hotspot: str) -> Dict[str, Any]:
        analysis_result = {"full_text": "N/A", "full_tokens": [], "full_surprisals": [], "hotspot_analysis": {},
                           "error": None}
        try:
            clean_context, clean_target = str(context).strip(), str(target).strip()
            analysis_result["full_text"] = f"{clean_context} {clean_target}"
            tokens, surprisals, offset_mapping = self.model_wrapper.get_surprisals(analysis_result["full_text"])
            analysis_result.update({"full_tokens": tokens, "full_surprisals": surprisals})

            stripped_hotspot = str(hotspot).strip()
            hotspot_char_start = clean_target.find(stripped_hotspot)
            if hotspot_char_start != -1:
                context_len = len(clean_context) + 1
                hotspot_char_start_full, hotspot_char_end_full = context_len + hotspot_char_start, context_len + hotspot_char_start + len(
                    stripped_hotspot)
                hotspot_indices = [i for i, (start, end) in enumerate(offset_mapping) if
                                   start < hotspot_char_end_full and end > hotspot_char_start_full]
                if hotspot_indices:
                    hotspot_surps = [surprisals[i] for i in hotspot_indices]
                    analysis_result["hotspot_analysis"] = {"avg_surprisal": np.mean(hotspot_surps).item(),
                                                           "sum_surprisal": np.sum(hotspot_surps).item(),
                                                           "num_tokens": len(hotspot_surps),
                                                           "tokens": [tokens[i] for i in hotspot_indices]}
        except Exception as e:
            analysis_result["error"] = str(e)
        return analysis_result

    def run(self, data: pd.DataFrame, source_filename: str) -> List[Dict]:
        if data.empty:
            logger.warning(f"Cannot run surprisal on empty dataframe for {source_filename}.")
            return []

        results = []
        for index, row in tqdm(data.iterrows(), total=len(data), desc=f"Surprisal for {source_filename}"):
            null_analysis = self._analyze_sentence(row["context"], row["null_sentence"], row["hotspot"])
            overt_analysis = self._analyze_sentence(row["context"], row["overt_sentence"], row["hotspot"])

            diff_score = None
            null_surp = null_analysis.get("hotspot_analysis", {}).get("avg_surprisal")
            overt_surp = overt_analysis.get("hotspot_analysis", {}).get("avg_surprisal")
            if null_surp is not None and overt_surp is not None:
                diff_score = null_surp - overt_surp

            results.append(
                {"item_id": row.get("item_id", index), "source_file": source_filename, "context": row["context"],
                 "hotspot_text": row["hotspot"], "null_sentence_analysis": null_analysis,
                 "overt_sentence_analysis": overt_analysis, "hotspot_difference_score": diff_score, })
        return results