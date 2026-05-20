Aplikasi Prediksi Keputusan Pembelian

Aplikasi Streamlit ini digunakan untuk menganalisis ulasan pelanggan dan memprediksi keputusan pembelian produk. Model prediksi menggunakan teks ulasan yang diproses dengan preprocessing Bahasa Indonesia, TF-IDF, serta dua algoritma klasifikasi: Naive Bayes sebagai model utama dan Random Forest sebagai model pembanding.

Dashboard juga menyediakan analisis rating dan price untuk mendukung tahap EDA, seperti distribusi rating, distribusi harga, korelasi, dan ringkasan per kelas keputusan pembelian.

Struktur Folder

```text
.
+-- app.py
+-- export_models.py
+-- feature_engineering.py
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

Jalankan ulang export setiap kali preprocessing, dataset, atau konfigurasi model berubah:

```bash
python export_models.py
```

Script akan:

- membaca dataset `.xlsx` jika tersedia, atau `.csv` jika tidak;
- memakai fitur teks `Review`/`Review_Tokenized`;
- melakukan case folding, cleaning, stopword removal, dan stemming;
- membentuk fitur TF-IDF dengan 5000 fitur unigram dan bigram;
- melatih Naive Bayes dan Random Forest;
- menyimpan model serta file evaluasi ke folder `models/`.

Menjalankan Aplikasi

```bash
streamlit run app.py
```

Menu Dashboard

- `Prediksi`: memasukkan ulasan, rating, dan price. Prediksi model menggunakan teks ulasan.
- `EDA Rating & Price`: menampilkan distribusi rating, distribusi price, korelasi, dan ringkasan per kelas keputusan.
- `Evaluasi Model`: menampilkan metrik, confusion matrix, classification report, dan cross-validation dari `models/evaluation_results.json`.

Model Utama

Naive Bayes digunakan sebagai model utama karena memberikan F1-score dan recall terbaik pada pipeline evaluasi berbasis TF-IDF. Random Forest disediakan sebagai pembanding untuk melihat perbedaan performa antar algoritma.
