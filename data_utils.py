"""Loading and cleaning the raw symptom dataset."""
from __future__ import annotations

import pandas as pd

from config import DATA_PATH


def _clean_symptom_list(raw: str) -> list[str]:
    """Split a comma-separated string into unique, lower-cased symptoms."""
    symptoms = {item.strip().lower() for item in str(raw).split(",")}
    return sorted(s for s in symptoms if s)


def load_disease_symptoms(path=DATA_PATH) -> pd.DataFrame:
    """Return one row per disease with the union of all its listed symptoms.

    The raw CSV has duplicate rows (e.g. the same disease listed several
    times), so rows are grouped by disease name and symptoms are merged.
    """
    df = pd.read_csv(path)
    df["disease"] = df["disease"].str.strip().str.lower()
    df["symptoms"] = df["symptoms"].apply(_clean_symptom_list)

    merged = (
        df.groupby("disease")["symptoms"]
        .apply(lambda rows: sorted({s for row in rows for s in row}))
        .reset_index()
    )
    return merged


def pretty(text: str) -> str:
    """Display-friendly version of a label (e.g. 'heart attack' -> 'Heart Attack')."""
    return text.title().replace("'S", "'s")
