# src/preprocess.py
"""
Basic preprocessing for Urdu QA dataset.
- Removes duplicates & nulls
- Normalizes text (whitespace, punctuation)
- Saves cleaned dataset to data/dataset_clean.csv
"""

import pandas as pd
import re
from src import config

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize Urdu punctuation variants
    text = text.replace("؟", "?").replace("٬", ",").replace("۔", ".")
    return text

def preprocess_dataset(input_path=config.DATASET_PATH, output_path=None):
    df = pd.read_csv(input_path)

    # Ensure required columns
    required = [config.COL_ID, config.COL_QUESTION, config.COL_ANSWER]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop nulls
    df = df.dropna(subset=[config.COL_QUESTION, config.COL_ANSWER])

    # Normalize text
    df[config.COL_QUESTION] = df[config.COL_QUESTION].apply(normalize_text)
    df[config.COL_ANSWER] = df[config.COL_ANSWER].apply(normalize_text)

    # Drop duplicates (keep first)
    df = df.drop_duplicates(subset=[config.COL_QUESTION])

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned dataset
    if output_path is None:
        output_path = config.DATA_DIR / "dataset_clean.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Preprocessed dataset saved to {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    config.ensure_dirs()
    preprocess_dataset()
