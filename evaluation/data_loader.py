# evaluation/data_loader.py (DEBUG Version)

import pandas as pd
from pathlib import Path
import logging

# Configure logger to be verbose for debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')


class DataLoader:
    """
    Handles loading and validation of surprisal evaluation data from CSV files.
    This version includes extensive logging to debug data loading issues.
    """
    INTERNAL_COLUMNS = ["item_id", "context", "null_sentence", "overt_sentence", "hotspot"]

    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at: {file_path}")
        self.file_path = file_path
        self.data = None

    def _get_language_suffix(self, df_cols: list) -> str:
        """Finds a common language suffix like '_italian' or '_english'."""
        for col in df_cols:
            if col.startswith('c_'):
                return col.split('c_', 1)[1]
        logger.warning(
            f"Could not determine a language suffix (e.g., '_italian') from 'c_' column in {self.file_path.name}. Assuming no suffix.")
        return ""

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV, validates it with verbose logging, and returns a standardized DataFrame.
        """
        logger.info(f"--- Loading Surprisal Data from: {self.file_path.name} ---")
        try:
            self.data = pd.read_csv(self.file_path).fillna('')
            self.data.columns = self.data.columns.str.strip()
            logger.info(f"Found columns: {list(self.data.columns)}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to read CSV file. Error: {e}")
            return pd.DataFrame()

        suffix = self._get_language_suffix(list(self.data.columns))

        # Define the columns we expect based on the detected suffix
        required_source_cols = {
            "item": "item_id",
            f"c_{suffix}": "context",
            f"t_null_{suffix}": "null_sentence",
            f"t_overt_{suffix}": "overt_sentence",
            f"hotspot_{suffix}": "hotspot",
        }
        logger.info(f"Expecting columns based on suffix '{suffix}': {list(required_source_cols.keys())}")

        # Validate that all required columns exist
        missing_cols = [col for col in required_source_cols if col not in self.data.columns]
        if missing_cols:
            logger.error(f"CRITICAL: File is missing required columns: {missing_cols}. Cannot process this file.")
            return pd.DataFrame()

        # Rename and select only the necessary columns
        renamed_data = self.data.rename(columns=required_source_cols)

        logger.info(f"Successfully loaded and mapped {len(renamed_data)} rows from '{self.file_path.name}'.")
        return renamed_data[self.INTERNAL_COLUMNS]