from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import preprocess_text


MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
NB_MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"


@st.cache_resource
def load_artifacts():
    missing = [
        str(path)
        for path in [VECTORIZER_PATH, RF_MODEL_PATH, NB_MODEL_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "File model belum ditemukan: "
            + ", ".join(missing)
            + ". Jalankan: python export_models.py"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    models = {
        "Random Forest": joblib.load(RF_MODEL_PATH),
        "Naive Bayes": joblib.load(NB_MODEL_PATH),
    }
    return vectorizer, models


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


st.set_page_config(
    page_title="Prediksi Keputusan Pembelian",
    layout="centered",
)

st.title("Prediksi Keputusan Pembelian")
st.write(
    "Aplikasi sederhana untuk memprediksi keputusan pembelian berdasarkan teks ulasan pelanggan."
)

try:
    vectorizer, model_options = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

example_reviews = {
    "Ulasan positif": "Rasanya enak, pengiriman cepat, packing aman, saya mau beli lagi.",
    "Ulasan negatif": "Barang kurang sesuai, rasanya tidak enak, pengiriman lama dan mengecewakan.",
    "Ulasan netral": "Produk sudah sampai, kemasan cukup baik, rasanya lumayan sesuai harga.",
}

selected_example = st.selectbox("Contoh input ulasan", list(example_reviews.keys()))
model_name = st.selectbox("Pilih model", list(model_options.keys()))

review_text = st.text_area(
    "Masukkan teks ulasan pelanggan",
    value=example_reviews[selected_example],
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

    st.subheader("Teks Setelah Preprocessing")
    st.code(processed_text or "(kosong)", language="text")
