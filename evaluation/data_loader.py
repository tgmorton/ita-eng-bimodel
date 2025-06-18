# evaluation/data_loader.py (Corrected and Final)

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and validation of surprisal evaluation data from CSV files.
    This version dynamically finds the required columns to be more robust.
    """
    # The generic column names we need internally
    INTERNAL_COLUMNS = [
        "item_id", "context", "null_sentence", "overt_sentence", "hotspot"
    ]

    def __init__(self, file_path: Path):
        """
        Initializes the DataLoader with the path to the evaluation file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at: {file_path}")
        self.file_path = file_path
        self.data = None

    def _map_columns(self) -> dict:
        """Dynamically find column mappings from the CSV header."""
        df_cols = [c.strip() for c in self.data.columns]
        mapping = {}

        # Map item and hotspot
        for col in df_cols:
            if col.lower() == 'item': mapping[col] = 'item_id'
            if 'hotspot' in col.lower(): mapping[col] = 'hotspot'

        # Find context, null, and overt columns
        for col in df_cols:
            if col.startswith('c_'):
                mapping[col] = 'context'
            elif col.startswith('t_null'):
                mapping[col] = 'null_sentence'
            elif col.startswith('t_overt'):
                mapping[col] = 'overt_sentence'

        return mapping

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV, validates it, and returns a DataFrame with standardized column names.
        """
        self.data = pd.read_csv(self.file_path).fillna('')
        column_mapping = self._map_columns()

        # Check if all internal columns were successfully mapped
        if len(set(column_mapping.values())) < len(self.INTERNAL_COLUMNS):
            mapped_cols = set(column_mapping.values())
            missing = set(self.INTERNAL_COLUMNS) - mapped_cols
            logger.warning(f"Could not find all required column types in {self.file_path.name}. Missing: {missing}")
            return pd.DataFrame()  # Return empty DataFrame if critical columns are missing

        # Rename and select only the necessary columns
        renamed_data = self.data.rename(columns=column_mapping)

        # Ensure all internal columns are present after renaming
        final_cols = [col for col in self.INTERNAL_COLUMNS if col in renamed_data.columns]

        logger.info(f"Successfully loaded and validated '{self.file_path.name}'.")
        return renamed_data[final_cols]