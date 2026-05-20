Aplikasi Prediksi Keputusan Pembelian

Aplikasi Streamlit ini memprediksi keputusan pembelian pelanggan berdasarkan teks ulasan. Pipeline mengikuti laporan lama/notebook baseline: preprocessing teks dengan stopword removal dan stemming Sastrawi, TF-IDF untuk ulasan, serta perbandingan Naive Bayes dan Random Forest.

Model utama yang digunakan untuk laporan dan dashboard adalah Naive Bayes. Random Forest tetap disediakan sebagai model pembanding.

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
    +-- evaluation_results.json
    +-- evaluation_results.csv
```

Install Library

```bash
pip install -r requirements.txt
```

Export Model dan Evaluasi

Jalankan ulang export setiap kali preprocessing, fitur, atau dataset berubah:

```bash
python export_models.py
```

Script akan:

- membaca dataset `.xlsx` jika tersedia, atau `.csv` jika tidak;
- memakai fitur teks `Review`/`Review_Tokenized`;
- melakukan stopword removal dan stemming;
- membentuk fitur TF-IDF dengan 5000 fitur unigram dan bigram;
- melatih Naive Bayes dan Random Forest;
- menyimpan model serta file evaluasi ke folder `models/`.

Menjalankan Aplikasi

```bash
streamlit run app.py
```

Dashboard memiliki dua halaman:

- `Prediksi`: memasukkan ulasan lalu memilih model.
- `Evaluasi Model`: menampilkan metrik, confusion matrix, classification report, dan cross-validation dari `models/evaluation_results.json`.

Catatan Laporan

Angka evaluasi di laporan PDF sebaiknya disamakan dengan isi terbaru `models/evaluation_results.json` setelah `python export_models.py` dijalankan.
