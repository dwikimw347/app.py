Catatan Revisi Laporan

Gunakan file `models/evaluation_results.json` sebagai sumber angka evaluasi terbaru setelah menjalankan:

```bash
python export_models.py
```

Bagian laporan yang perlu diselaraskan:

1. Metodologi
   - Fitur input model: `Review`, `Rating`, dan `price`.
   - Preprocessing teks: case folding, cleaning URL/simbol/angka, tokenizing, stopword removal, dan stemming Sastrawi.
   - Ekstraksi fitur: TF-IDF untuk teks, MinMax scaling untuk `rating` dan `log1p(price)`.
   - Penanganan imbalance: SMOTE diterapkan hanya pada training set setelah split 80:20.

2. Pemodelan
   - Random Forest menjadi model utama berdasarkan evaluasi pipeline revisi.
   - Naive Bayes menjadi model pembanding.
   - Enam varian Naive Bayes dan enam konfigurasi Random Forest tetap dievaluasi.

3. Hasil dan Pembahasan
   - Tabel eksperimen, classification report, confusion matrix, dan cross-validation harus mengambil nilai dari `models/evaluation_results.json`.
   - Hindari mencampur angka lama dari notebook sebelum revisi fitur rating, price, stopword removal, dan SMOTE.

4. Dashboard
   - Dashboard Streamlit sekarang memiliki halaman `Prediksi` dan `Evaluasi Model`.
   - Halaman prediksi memakai input ulasan, rating, dan price.
   - Halaman evaluasi membaca metrik dari `models/evaluation_results.json`.
