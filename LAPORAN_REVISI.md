Catatan Revisi Laporan

Gunakan file `models/evaluation_results.json` sebagai sumber angka evaluasi terbaru setelah menjalankan:

```bash
python export_models.py
```

Bagian laporan yang perlu diselaraskan:

1. Metodologi
   - Fitur input model: teks `Review` atau `Review_Tokenized`.
   - Preprocessing teks: case folding, cleaning URL/simbol/angka, tokenizing, stopword removal, dan stemming Sastrawi.
   - Ekstraksi fitur: TF-IDF untuk teks.
   - Penanganan imbalance: Random Forest memakai `class_weight='balanced'` sebagai pembanding.

2. Pemodelan
   - Naive Bayes menjadi model utama sesuai laporan lama.
   - Random Forest menjadi model pembanding.
   - Enam varian Naive Bayes dan enam konfigurasi Random Forest tetap dievaluasi.

3. Hasil dan Pembahasan
   - Tabel eksperimen, classification report, confusion matrix, dan cross-validation harus mengambil nilai dari `models/evaluation_results.json`.
   - Hindari mencampur angka dari pipeline revisi rating/price/SMOTE dengan baseline laporan lama.

4. Dashboard
   - Dashboard Streamlit sekarang memiliki halaman `Prediksi` dan `Evaluasi Model`.
   - Halaman prediksi memakai input ulasan.
   - Halaman evaluasi membaca metrik dari `models/evaluation_results.json`.
