import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


NUMERIC_FEATURE_NAMES = ["rating", "price_log"]


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


def make_numeric_frame(ratings, prices) -> pd.DataFrame:
    rating = pd.Series(ratings).apply(parse_rating).clip(lower=1, upper=5)
    price = pd.Series(prices).apply(parse_price).clip(lower=0)
    return pd.DataFrame(
        {
            "rating": rating,
            "price_log": np.log1p(price),
        }
    )


def build_feature_matrix(
    texts,
    ratings,
    prices,
    vectorizer,
    scaler,
    fit_vectorizer=False,
    fit_scaler=False,
):
    if fit_vectorizer:
        text_features = vectorizer.fit_transform(texts)
    else:
        text_features = vectorizer.transform(texts)

    numeric_frame = make_numeric_frame(ratings, prices)
    if numeric_frame.isna().any().any():
        raise ValueError("Rating dan price wajib valid untuk membentuk fitur model.")

    if fit_scaler:
        numeric_features = scaler.fit_transform(numeric_frame)
    else:
        numeric_features = scaler.transform(numeric_frame)

    return hstack([text_features, csr_matrix(numeric_features)], format="csr")
