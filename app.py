import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import preprocess_text


MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
NB_MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"
EVALUATION_PATH = MODEL_DIR / "evaluation_results.json"


@st.cache_resource
def load_artifacts():
    required_paths = [VECTORIZER_PATH, RF_MODEL_PATH, NB_MODEL_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefak model belum lengkap: "
            + ", ".join(missing)
            + ". Jalankan ulang: python export_models.py"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    models = {
        "Naive Bayes (Model Utama)": joblib.load(NB_MODEL_PATH),
        "Random Forest (Pembanding)": joblib.load(RF_MODEL_PATH),
    }
    return vectorizer, models


@st.cache_data
def load_evaluation():
    if not EVALUATION_PATH.exists():
        return None
    return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))


def prediction_label(value) -> str:
    clean = str(value).strip().upper()
    if clean in {"1", "Y", "YA", "YES", "BELI", "TRUE"}:
        return "BELI"
    return "TIDAK BELI"


def show_probability(model, features):
    if not hasattr(model, "predict_proba"):
        st.info("Model ini tidak menyediakan probabilitas prediksi.")
        return

    probabilities = model.predict_proba(features)[0]
    classes = list(getattr(model, "classes_", range(len(probabilities))))
    rows = []
    for class_value, probability in zip(classes, probabilities):
        rows.append(
            {
                "Kelas": prediction_label(class_value),
                "Probabilitas": f"{probability * 100:.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def format_metric(value):
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def render_prediction_page(vectorizer, model_options):
    st.title("Prediksi Keputusan Pembelian")

    example_reviews = {
        "Ulasan positif": {
            "review": "Rasanya enak, pengiriman cepat, packing aman, saya mau beli lagi.",
        },
        "Ulasan negatif": {
            "review": "Barang kurang sesuai, rasanya tidak enak, pengiriman lama dan mengecewakan.",
        },
        "Ulasan netral": {
            "review": "Produk sudah sampai, kemasan cukup baik, rasanya lumayan sesuai harga.",
        },
    }

    selected_example = st.selectbox("Contoh input ulasan", list(example_reviews.keys()))
    example = example_reviews[selected_example]

    model_name = st.selectbox("Pilih model", list(model_options.keys()))
    review_text = st.text_area(
        "Masukkan teks ulasan pelanggan",
        value=example["review"],
        height=150,
    )

    if st.button("Prediksi", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Masukkan teks ulasan terlebih dahulu.")
            st.stop()

        processed_text = preprocess_text(review_text)
        features = vectorizer.transform([processed_text])
        model = model_options[model_name]
        prediction = model.predict(features)[0]
        result = prediction_label(prediction)

        if result == "BELI":
            st.success(f"Hasil prediksi: {result}")
        else:
            st.error(f"Hasil prediksi: {result}")

        st.subheader("Probabilitas Prediksi")
        show_probability(model, features)

        st.subheader("Fitur yang Digunakan")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Review setelah preprocessing": processed_text or "(kosong)",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def best_results_table(evaluation):
    rows = []
    for family, result in evaluation["best_results"].items():
        cv = result.get("cross_validation", {})
        rows.append(
            {
                "Model": family,
                "Varian Terbaik": result["model"],
                "Accuracy": format_metric(result["accuracy"]),
                "Precision": format_metric(result["precision"]),
                "Recall": format_metric(result["recall"]),
                "F1-Score": format_metric(result["f1_score"]),
                "AUC-ROC": format_metric(result["auc_roc"]),
                "CV F1 Mean": format_metric(cv.get("mean")),
                "CV F1 Std": format_metric(cv.get("std")),
            }
        )
    return pd.DataFrame(rows)


def experiments_table(evaluation):
    rows = []
    for result in evaluation["all_experiments"]:
        rows.append(
            {
                "Model": result["family"],
                "Varian": result["model"],
                "Accuracy": format_metric(result["accuracy"]),
                "Precision": format_metric(result["precision"]),
                "Recall": format_metric(result["recall"]),
                "F1-Score": format_metric(result["f1_score"]),
                "AUC-ROC": format_metric(result["auc_roc"]),
            }
        )
    return pd.DataFrame(rows)


def render_model_detail(result):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", format_metric(result["accuracy"]))
    col2.metric("Precision", format_metric(result["precision"]))
    col3.metric("Recall", format_metric(result["recall"]))
    col4.metric("F1-Score", format_metric(result["f1_score"]))
    col5.metric("AUC-ROC", format_metric(result["auc_roc"]))

    cm = result["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Aktual: Tidak Beli", "Aktual: Beli"],
        columns=["Pred: Tidak Beli", "Pred: Beli"],
    )
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    report = pd.DataFrame(result["classification_report"]).T
    st.subheader("Classification Report")
    st.dataframe(report, use_container_width=True)

    cv = result.get("cross_validation")
    if cv:
        st.subheader("Cross-Validation 5-Fold")
        st.dataframe(
            pd.DataFrame(
                {
                    "Fold": [f"Fold {index}" for index in range(1, len(cv["scores"]) + 1)],
                    "F1-Score": [format_metric(score) for score in cv["scores"]],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_evaluation_page():
    st.title("Evaluasi Model")
    evaluation = load_evaluation()
    if evaluation is None:
        st.warning(
            "File evaluasi belum tersedia. Jalankan `python export_models.py` "
            "untuk membuat `models/evaluation_results.json`."
        )
        return

    st.info(f"Model utama yang ditegaskan untuk laporan: {evaluation['main_model']}")
    st.caption(f"Evaluasi dibuat pada: {evaluation['generated_at']}")

    st.subheader("Ringkasan Model Terbaik")
    st.dataframe(best_results_table(evaluation), hide_index=True, use_container_width=True)

    st.subheader("Seluruh Eksperimen")
    st.dataframe(experiments_table(evaluation), hide_index=True, use_container_width=True)

    selected_model = st.selectbox(
        "Detail model terbaik",
        list(evaluation["best_results"].keys()),
    )
    render_model_detail(evaluation["best_results"][selected_model])


st.set_page_config(
    page_title="Prediksi Keputusan Pembelian",
    layout="wide",
)

try:
    vectorizer, model_options = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

page = st.sidebar.radio("Menu", ["Prediksi", "Evaluasi Model"])

if page == "Prediksi":
    render_prediction_page(vectorizer, model_options)
else:
    render_evaluation_page()
