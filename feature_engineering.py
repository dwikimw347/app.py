import re

import numpy as np
import pandas as pd


def parse_rating(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0))


def parse_price(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    text = text.replace("rp", "").replace("idr", "").replace(" ", "")

    if "," in text:
        normalized = text.replace(".", "").replace(",", ".")
        try:
            return float(normalized)
        except ValueError:
            pass

    if "." in text:
        parts = text.split(".")
        if all(part.isdigit() for part in parts) and len(parts[-1]) == 3:
            return float("".join(parts))
        try:
            return float(text)
        except ValueError:
            pass

    digits = re.sub(r"\D", "", text)
    return float(digits) if digits else np.nan
