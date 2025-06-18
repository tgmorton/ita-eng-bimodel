# evaluation/data_loader.py (Final, Language-Specific Version)

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and validation of surprisal evaluation data from CSV files
    for a specific language.
    """
    INTERNAL_COLUMNS = ["item_id", "context", "null_sentence", "overt_sentence", "hotspot"]

    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at: {file_path}")
        self.file_path = file_path

    def get_available_languages(self) -> list:
        """Inspects the CSV header to find which languages are available."""
        try:
            df_cols = pd.read_csv(self.file_path, nrows=0).columns
            languages = set()
            for col in df_cols:
                if col.startswith('c_'):
                    languages.add(col.split('c_', 1)[1])
            return sorted(list(languages))
        except Exception as e:
            logger.error(f"Could not read headers from {self.file_path.name}: {e}")
            return []

    def load_data(self, language: str) -> pd.DataFrame:
        """
        Loads the CSV for a specific language, validates it, and returns a
        DataFrame with standardized column names.
        """
        logger.info(f"Attempting to load '{language}' columns from: {self.file_path.name}")
        try:
            df = pd.read_csv(self.file_path).fillna('')
            df.columns = df.columns.str.strip()
        except Exception as e:
            logger.error(f"Failed to read CSV file. Error: {e}")
            return pd.DataFrame()

        # Define the exact columns required for the specified language
        required_source_cols = {
            "item": "item_id",
            f"c_{language}": "context",
            f"t_null_{language}": "null_sentence",
            f"t_overt_{language}": "overt_sentence",
            f"hotspot_{language}": "hotspot",
        }

        missing_cols = [col for col in required_source_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"File '{self.file_path.name}' is missing required '{language}' columns: {missing_cols}")
            return pd.DataFrame()

        renamed_data = df.rename(columns=required_source_cols)

        logger.info(f"Successfully loaded and mapped {len(renamed_data)} rows for '{language}'.")
        return renamed_data[self.INTERNAL_COLUMNS]