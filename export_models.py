import ast
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from preprocessing import preprocess_text

DATASET_XLSX = Path("preprocessed_data_ulasan.xlsx")
DATASET_CSV = Path("preprocessed_data_ulasan.csv")
MODEL_DIR = Path("models")

TEXT_COLUMN_CANDIDATES = [
    "Review_Stemmed",
    "review_stemmed",
    "teks_bersih",
    "Review_Tokenized",
    "review_tokenized",
    "Review",
    "review",
]
TARGET_COLUMN_CANDIDATES = [
    "purchase_decision",
    "Purchase_Decision",
    "target",
    "label",
    "Label",
    "keputusan_pembelian",
]


def parse_custom_review_csv(path: Path) -> pd.DataFrame:
    """Fallback parser for the project CSV, where reviews can contain commas/newlines."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header, body = text.split("\n", 1)
    expected_header = "Review,Rating,price,purchase_decision,Review_Tokenized"
    if header.strip() != expected_header:
        raise ValueError(
            "CSV tidak bisa dibaca otomatis. Kolom yang ditemukan: "
            f"{header.strip()}. Format yang diharapkan: {expected_header}"
        )

    records = []
    buffer = []
    for line in body.splitlines():
        buffer.append(line)
        if line.endswith('"""'):
            records.append("\n".join(buffer))
            buffer = []

    pattern = re.compile(
        r'^"?(?P<review>.*),(?P<rating>\d+(?:\.\d+)?),'
        r'(?P<price>\d+(?:\.\d+)?),(?P<label>[^,]+),'
        r'""(?P<tokens>\[.*\])"""$',
        re.DOTALL,
    )

    rows = []
    failed = 0
    for record in records:
        match = pattern.match(record)
        if not match:
            failed += 1
            continue
        rows.append(
            {
                "Review": match.group("review").strip(),
                "Rating": float(match.group("rating")),
                "price": float(match.group("price")),
                "purchase_decision": match.group("label").strip(),
                "Review_Tokenized": match.group("tokens"),
            }
        )

    if failed:
        print(f"Peringatan: {failed} baris CSV gagal diparse dan dilewati.")
    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    if DATASET_XLSX.exists():
        return pd.read_excel(DATASET_XLSX)

    if DATASET_CSV.exists():
        try:
            return pd.read_csv(DATASET_CSV)
        except Exception:
            return parse_custom_review_csv(DATASET_CSV)

    raise FileNotFoundError(
        "Dataset tidak ditemukan. Letakkan salah satu file berikut di folder project: "
        f"{DATASET_XLSX.name} atau {DATASET_CSV.name}"
    )


def find_column(columns, candidates, role):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Kolom {role} tidak ditemukan.\n"
        f"Kolom yang tersedia: {list(columns)}\n"
        f"Kandidat yang dicari: {candidates}"
    )


def list_to_sentence(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, list):
        return " ".join(str(token) for token in value)
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(token) for token in parsed)
    except Exception:
        pass
    return text


def normalize_label(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.strip().str.upper()
    positive_values = {"Y", "YA", "YES", "BELI", "1", "TRUE"}
    return clean.isin(positive_values).astype(int)


def train_best_random_forest(X_train, X_test, y_train, y_test):
    configs = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
    ]

    best_model = None
    best_f1 = -1
    best_config = None
    for config in configs:
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        score = f1_score(y_test, model.predict(X_test), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_config = config

    print(f"Random Forest terbaik: {best_config}, F1={best_f1:.4f}")
    return best_model


def train_best_naive_bayes(X_train, X_test, y_train, y_test):
    candidates = {
        "Multinomial NB (alpha=0.1)": MultinomialNB(alpha=0.1),
        "Multinomial NB (alpha=0.5)": MultinomialNB(alpha=0.5),
        "Multinomial NB (alpha=1.0)": MultinomialNB(alpha=1.0),
        "Complement NB (alpha=0.1)": ComplementNB(alpha=0.1),
        "Complement NB (alpha=0.5)": ComplementNB(alpha=0.5),
        "Complement NB (alpha=1.0)": ComplementNB(alpha=1.0),
    }

    best_model = None
    best_f1 = -1
    best_name = None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        score = f1_score(y_test, model.predict(X_test), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_name = name

    print(f"Naive Bayes terbaik: {best_name}, F1={best_f1:.4f}")
    return best_model


def main():
    df = load_dataset()
    print(f"Dataset terbaca: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Kolom ditemukan: {list(df.columns)}")

    text_column = find_column(df.columns, TEXT_COLUMN_CANDIDATES, "teks ulasan")
    target_column = find_column(df.columns, TARGET_COLUMN_CANDIDATES, "label/target")
    print(f"Kolom teks ulasan: {text_column}")
    print(f"Kolom label target: {target_column}")
    print("Format label target:")
    print(df[target_column].astype(str).str.strip().value_counts(dropna=False).to_string())

    df = df.copy()
    df["teks_bersih"] = df[text_column].apply(list_to_sentence).apply(preprocess_text)
    df = df[df["teks_bersih"].str.strip() != ""].reset_index(drop=True)
    y = normalize_label(df[target_column])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["teks_bersih"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    random_forest = train_best_random_forest(X_train, X_test, y_train, y_test)
    naive_bayes = train_best_naive_bayes(X_train, X_test, y_train, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(random_forest, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(naive_bayes, MODEL_DIR / "naive_bayes_model.pkl")

    print("Model berhasil disimpan:")
    print(f"- {MODEL_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"- {MODEL_DIR / 'random_forest_model.pkl'}")
    print(f"- {MODEL_DIR / 'naive_bayes_model.pkl'}")


if __name__ == "__main__":
    main()
