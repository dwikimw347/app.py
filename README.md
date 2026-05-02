Aplikasi Prediksi Keputusan Pembelian

Aplikasi Streamlit ini memprediksi keputusan pembelian pelanggan berdasarkan teks ulasan. Model menggunakan preprocessing teks dengan stemming Sastrawi, TF-IDF, dan dua pilihan classifier: Random Forest sebagai model utama dan Naive Bayes sebagai pembanding.

Struktur Folder

```text
.
+-- app.py
+-- export_models.py
+-- preprocessing.py
+-- requirements.txt
+-- README.md
+-- preprocessed_data_ulasan.csv
+-- TF_IDF_Split_Data.ipynb
+-- Model random forest trained.ipynb
+-- Model_Naive_Bayes _Trained.ipynb
+-- models/
    +-- tfidf_vectorizer.pkl
    +-- random_forest_model.pkl
    +-- naive_bayes_model.pkl
```

Install Library

```bash
pip install -r requirements.txt
```

## Export Model

Jika file `.pkl` belum ada di folder `models/`, jalankan:

```bash
python export_models.py
```

Script akan membaca dataset `preprocessed_data_ulasan.xlsx` jika tersedia. Jika tidak ada, script akan memakai `preprocessed_data_ulasan.csv`. Kolom teks dan label akan dicek otomatis, lalu model disimpan ke folder `models/`.

## Menjalankan Aplikasi

```bash
streamlit run app.py
atau
py -m streamlit run app.py

```

Setelah aplikasi terbuka, pilih model, masukkan ulasan pelanggan, lalu klik tombol **Prediksi**.
