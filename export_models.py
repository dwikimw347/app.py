import ast
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from preprocessing import preprocess_text


DATASET_XLSX = Path("preprocessed_data_ulasan.xlsx")
DATASET_CSV = Path("preprocessed_data_ulasan.csv")
MODEL_DIR = Path("models")

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
NB_MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"
EVALUATION_JSON_PATH = MODEL_DIR / "evaluation_results.json"
EVALUATION_CSV_PATH = MODEL_DIR / "evaluation_results.csv"

MAIN_MODEL = "Naive Bayes"

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


def positive_scores(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return model.predict_proba(X)[:, positive_index]


def evaluate_model(family, model_name, parameters, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=labels).ravel()
    return {
        "family": family,
        "model": model_name,
        "parameters": parameters,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_score) if y_score is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=["Tidak Beli (N)", "Beli (Y)"],
            output_dict=True,
            zero_division=0,
        ),
    }


def cross_validate_model(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )
    return {
        "scores": [float(score) for score in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


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
    best_result = None
    results = []
    for config in configs:
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        depth = "None" if config["max_depth"] is None else str(config["max_depth"])
        model_name = f"RF (n={config['n_estimators']}, depth={depth})"
        result = evaluate_model("Random Forest", model_name, config, model, X_test, y_test)
        results.append(result)
        if best_result is None or result["f1_score"] > best_result["f1_score"]:
            best_model = model
            best_result = result

    best_result["cross_validation"] = cross_validate_model(best_model, X_train, y_train)
    print(
        "Random Forest terbaik: "
        f"{best_result['model']}, F1={best_result['f1_score']:.4f}, "
        f"AUC={best_result['auc_roc']:.4f}"
    )
    return best_model, best_result, results


def train_best_naive_bayes(X_train, X_test, y_train, y_test):
    candidates = [
        ("Multinomial NB (alpha=0.1)", {"variant": "MultinomialNB", "alpha": 0.1}, MultinomialNB(alpha=0.1)),
        ("Multinomial NB (alpha=0.5)", {"variant": "MultinomialNB", "alpha": 0.5}, MultinomialNB(alpha=0.5)),
        ("Multinomial NB (alpha=1.0)", {"variant": "MultinomialNB", "alpha": 1.0}, MultinomialNB(alpha=1.0)),
        ("Complement NB (alpha=0.1)", {"variant": "ComplementNB", "alpha": 0.1}, ComplementNB(alpha=0.1)),
        ("Complement NB (alpha=0.5)", {"variant": "ComplementNB", "alpha": 0.5}, ComplementNB(alpha=0.5)),
        ("Complement NB (alpha=1.0)", {"variant": "ComplementNB", "alpha": 1.0}, ComplementNB(alpha=1.0)),
    ]

    best_model = None
    best_result = None
    results = []
    for name, parameters, model in candidates:
        model.fit(X_train, y_train)
        result = evaluate_model("Naive Bayes", name, parameters, model, X_test, y_test)
        results.append(result)
        if best_result is None or result["f1_score"] > best_result["f1_score"]:
            best_model = model
            best_result = result

    best_result["cross_validation"] = cross_validate_model(best_model, X_train, y_train)
    print(
        "Naive Bayes terbaik: "
        f"{best_result['model']}, F1={best_result['f1_score']:.4f}, "
        f"AUC={best_result['auc_roc']:.4f}"
    )
    return best_model, best_result, results


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_evaluation(best_results, all_results, dataset_info):
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "main_model": MAIN_MODEL,
        "feature_set": ["Review TF-IDF"],
        "analysis_features": ["Rating", "price"],
        "preprocessing": [
            "case folding",
            "cleaning URL/simbol/angka",
            "tokenizing",
            "stopword removal dengan negasi dipertahankan",
            "stemming Sastrawi",
        ],
        "split": {"train": "80%", "test": "20%", "random_state": 42},
        "imbalance_handling": {
            "method": "class_weight='balanced' pada Random Forest",
            "applied_to": "Random Forest sebagai pembanding",
        },
        "dataset": dataset_info,
        "best_results": best_results,
        "all_experiments": all_results,
    }
    payload = to_builtin(payload)
    EVALUATION_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for result in all_results:
        rows.append(
            {
                "family": result["family"],
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "auc_roc": result["auc_roc"],
                "tn": result["confusion_matrix"]["tn"],
                "fp": result["confusion_matrix"]["fp"],
                "fn": result["confusion_matrix"]["fn"],
                "tp": result["confusion_matrix"]["tp"],
            }
        )
    pd.DataFrame(rows).to_csv(EVALUATION_CSV_PATH, index=False)


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

    prepared = df.copy()
    prepared["teks_bersih"] = prepared[text_column].apply(list_to_sentence).apply(preprocess_text)
    prepared["target"] = normalize_label(prepared[target_column])
    prepared = prepared[prepared["teks_bersih"].str.strip() != ""].reset_index(drop=True)
    y = prepared["target"]

    print(f"Baris valid setelah preprocessing: {prepared.shape[0]:,}")
    print("Distribusi target setelah normalisasi:")
    print(y.value_counts().rename(index={0: "TIDAK BELI", 1: "BELI"}).to_string())

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(prepared["teks_bersih"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Ukuran training set: {X_train.shape}")
    print(f"Ukuran testing set: {X_test.shape}")

    random_forest, rf_best_result, rf_results = train_best_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    naive_bayes, nb_best_result, nb_results = train_best_naive_bayes(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(random_forest, RF_MODEL_PATH)
    joblib.dump(naive_bayes, NB_MODEL_PATH)

    dataset_info = {
        "source_rows": int(df.shape[0]),
        "valid_rows": int(prepared.shape[0]),
        "text_column": text_column,
        "target_column": target_column,
        "matrix_shape": list(X.shape),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }
    save_evaluation(
        best_results={
            "Naive Bayes": nb_best_result,
            "Random Forest": rf_best_result,
        },
        all_results=nb_results + rf_results,
        dataset_info=dataset_info,
    )

    print("Model dan evaluasi berhasil disimpan:")
    print(f"- {VECTORIZER_PATH}")
    print(f"- {NB_MODEL_PATH} ({MAIN_MODEL}, model utama)")
    print(f"- {RF_MODEL_PATH} (model pembanding)")
    print(f"- {EVALUATION_JSON_PATH}")
    print(f"- {EVALUATION_CSV_PATH}")


if __name__ == "__main__":
    main()
