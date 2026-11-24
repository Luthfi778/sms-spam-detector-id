# 🛡️ Detektor SMS Spam & Penipuan Berbahasa Indonesia

Aplikasi web (Streamlit) yang mengklasifikasikan pesan SMS/WhatsApp berbahasa Indonesia sebagai **Spam** atau **Normal**, lengkap dengan halaman evaluasi model (precision, recall, F1, confusion matrix) dan analisis kesalahan (false positive / false negative).

Proyek ini dibuat untuk **Final Project Keamanan Informasi, Topik 1: Defensive AI** (Program Studi Informatika, Universitas Siliwangi). Sistem berfungsi sebagai deteksi dini ancaman *social engineering* lewat pesan singkat (smishing).

---

## Daftar Isi

1. [Fitur](#fitur)
2. [Struktur proyek](#struktur-proyek)
3. [Dataset](#dataset)
4. [Metode](#metode)
5. [Hasil evaluasi](#hasil-evaluasi)
6. [Instalasi dan menjalankan](#instalasi-dan-menjalankan)
7. [Opsional: IndoBERT di Google Colab](#opsional-indobert-di-google-colab)
8. [Keterbatasan](#keterbatasan)
9. [Etika dan privasi](#etika-dan-privasi)
10. [Pemetaan ke luaran tugas](#pemetaan-ke-luaran-tugas)
11. [Pemecahan masalah](#pemecahan-masalah)
12. [Atribusi](#atribusi)

---

## Fitur

| Halaman | Isi |
|---|---|
| **Deteksi** (`app.py`) | Tempel pesan, pilih model, atur ambang Spam. Menampilkan label, probabilitas, teks setelah preprocessing, dan kata-kata yang paling memengaruhi keputusan. Ada 5 contoh pesan siap klik untuk demo. |
| **Evaluasi Model** | Tabel perbandingan 2 model (Naive Bayes vs Logistic Regression: accuracy, precision, recall, F1, PR-AUC, ROC-AUC, waktu latih, waktu inferensi), confusion matrix tiap model, kurva Precision–Recall. |
| **Analisis Kesalahan** | Daftar false positive dan false negative per model, pesan yang salah di semua model, serta uji ketahanan terhadap pesan yang disamarkan penipu. |
| **Tentang Data** | Sumber, proses pembersihan, distribusi kelas, panjang pesan, kata teratas per kelas. |

## Struktur proyek

```
sms-spam-detector-id/
├── app.py                        # Halaman Deteksi (entry point Streamlit)
├── pages/
│   ├── 1_Evaluasi_Model.py
│   ├── 2_Analisis_Kesalahan.py
│   └── 3_Tentang_Data.py
├── src/
│   ├── config.py                 # Path, seed, nama model
│   ├── preprocessing.py          # clean_text (dipakai training DAN aplikasi)
│   ├── data.py                   # Load, hapus duplikat, split
│   ├── train.py                  # Training + evaluasi + uji ketahanan
│   ├── evaluation.py             # Perhitungan metrik
│   ├── explain.py                # Penjelasan kata (Logistic Regression)
│   ├── robustness.py             # Simulasi pesan yang disamarkan
│   └── app_utils.py              # Loader artefak untuk Streamlit
├── scripts/
│   └── train_indobert.py         # OPSIONAL, untuk Colab + GPU
├── data/
│   └── indo_spam.csv             # Dataset mentah
├── artifacts/                    # Hasil training (sudah disertakan)
│   ├── models/*.joblib
│   ├── metrics.json
│   ├── test_predictions.csv
│   ├── robustness.csv
│   ├── robustness_examples.csv
│   ├── data_summary.json
│   ├── dataset_clean.csv
│   └── versions.json             # versi scikit-learn saat training
├── tests/test_preprocessing.py
├── requirements.txt
├── requirements-indobert.txt
└── requirements-dev.txt
```

Prinsip desain: **training dan aplikasi dipisah**. `src/train.py` melatih model dan menyimpan hasilnya ke `artifacts/`. Aplikasi web hanya memuat file-file itu, jadi cepat dibuka dan tidak melatih ulang. Pipeline yang disimpan sudah mencakup preprocessing, sehingga hasil di web identik dengan hasil evaluasi.

## Dataset

- **Nama:** Indonesian SMS Spam Detection
- **Sumber:** Kaggle; berkas `indo_spam.csv` yang sama diambil dari `https://raw.githubusercontent.com/gevabriel/dataset/main/indo_spam.csv` (dipakai juga oleh notebook referensi *Indonesian SMS Spam Detection using IndoBERT*).
- **Kolom:** `Kategori` (`ham` / `Spam`) dan `Pesan` (teks SMS).
- **Ukuran mentah:** 2.045 pesan (1.302 ham, 743 Spam).

### Pembersihan data

| Tahap | Jumlah |
|---|---|
| Data mentah | 2.045 |
| Duplikat dibuang (pesan sama setelah preprocessing) | 186 |
| Pesan kosong / label bertentangan | 0 / 0 |
| **Data bersih** | **1.859** (Normal 1.257, Spam 602, Spam ≈ 32,4%) |
| Train / Test (stratified 80/20, seed 42) | 1.487 / 372 |

Duplikat dibuang **sebelum** pembagian data. Jika tidak, pesan yang sama bisa muncul di data latih dan data uji, sehingga skor tampak lebih bagus dari kenyataan.

> ⚠️ **Catatan penting tentang dataset.** Isi pesan (nomor shortcode seperti "87121", "ringtone", mata uang pound, nama-nama Inggris) menunjukkan bahwa dataset ini tampaknya **terjemahan dari UCI SMS Spam Collection** (berbahasa Inggris), bukan SMS asli Indonesia. Ini kesimpulan dari membaca isi data, bukan dari dokumentasi resmi, jadi sebaiknya dikonfirmasi di halaman Kaggle. Konsekuensinya: model lebih mengenali pola spam gaya Inggris yang diterjemahkan daripada modus penipuan lokal (pinjol ilegal, kurir palsu, "mama minta pulsa"). Bahas ini di laporan sebagai keterbatasan.

## Metode

**Preprocessing** (`src/preprocessing.py`):
- Huruf kecil; simbol dibuang.
- URL → token `urltoken`, nominal uang ("Rp 75 juta", "£1000") → `uang`, angka panjang (nomor telepon, shortcode) → `nomorpanjang`. Ini **tidak dibuang** karena merupakan sinyal kuat spam.
- Normalisasi penyamaran: `G R A T I S`, `g.r.a.t.i.s`, dan `h4d14h` dikembalikan menjadi `gratis` dan `hadiah`.
- Kamus kecil bahasa gaul (`yg`→`yang`, `gak`→`tidak`, dst.).
- Tidak memakai stemming agar tidak menambah dependensi; bisa ditambahkan dengan Sastrawi jika diperlukan.

**Fitur:** TF-IDF kata (1–2 gram) digabung TF-IDF karakter (3–5 gram, lebih tahan typo).

**Model** (hyperparameter dipilih dengan GridSearchCV 5-lipat pada data latih, skor F1 macro; data uji hanya dipakai sekali di akhir):

| Model | Catatan |
|---|---|
| Multinomial Naive Bayes | Baseline klasik untuk teks |
| Logistic Regression | `class_weight=balanced`; dipakai juga untuk penjelasan kata |

Dua model dipilih karena paling mudah dijelaskan: Naive Bayes menjumlahkan bukti kata per kelas (baseline), sedangkan Logistic Regression memberi bobot tiap kata dan bobot itu bisa ditampilkan sebagai penjelasan prediksi.

## Hasil evaluasi

Dihitung pada data uji (372 pesan: 252 Normal, 120 Spam). Kelas positif = **Spam**. Angka ini dari satu kali run dengan seed 42; jalankan `python -m src.train` untuk mereproduksi.

| Model | Accuracy | Precision | Recall | F1 | F1 macro | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Naive Bayes | 0,970 | 0,966 | 0,942 | 0,954 | 0,966 | 0,976 | 4 | 7 |
| **Logistic Regression** | **0,973** | 0,974 | 0,942 | **0,958** | **0,969** | **0,987** | 3 | 7 |

(Precision, Recall, F1 untuk kelas Spam; FP = Normal salah jadi Spam; FN = Spam lolos.)

**Cara membaca hasil ini:**
- Logistic Regression sedikit lebih baik daripada Naive Bayes (F1 0,958 vs 0,954; PR-AUC 0,987 vs 0,976). Selisihnya hanya 1 pesan false positive, jadi **bukan perbedaan yang bermakna secara statistik** pada data uji sekecil ini. Keduanya sama-sama meloloskan 7 spam (recall 0,942).
- Karena hasilnya hampir sama, pilihan model bisa didasarkan pada hal lain: Logistic Regression dapat menjelaskan kata mana yang memengaruhi keputusan, sedangkan Naive Bayes lebih sederhana dan sangat cepat.
- Beberapa kesalahan tampak berasal dari **label yang meragukan** (mis. pesan yang jelas normal berlabel Spam). Ada 9 pesan yang salah diprediksi oleh kedua model; periksa di halaman Analisis Kesalahan.
- Skor tinggi sebagian disebabkan dataset yang relatif "mudah" dan tampaknya terjemahan (lihat catatan dataset). Jangan menyimpulkan bahwa model akan sebaik ini pada SMS penipuan Indonesia yang sebenarnya.

### Uji ketahanan (recall Spam pada pesan yang disamarkan)

Pesan Spam di data uji (120 pesan) diubah seperti trik penipu, lalu diukur recall-nya.

| Model | Asli | Spasi antar huruf | Titik antar huruf | Leetspeak | Typo (hapus huruf) |
|---|---|---|---|---|---|
| Naive Bayes | 0,942 | 0,942 | 0,942 | 0,950 | 0,942 |
| Logistic Regression | 0,942 | 0,942 | 0,942 | 0,942 | 0,933 |
| Logistic Regression **tanpa normalisasi** | 0,925 | 0,900 | 0,900 | 0,892 | 0,917 |

Normalisasi penyamaran di preprocessing menjaga recall tetap stabil, sedangkan model pembanding tanpa normalisasi turun sekitar 2–3 poin dari skor aslinya (≈ 3–4 pesan). Naive Bayes naik 1 pesan pada leetspeak; itu fluktuasi, bukan peningkatan nyata. Sampel kecil dan ini simulasi sederhana, **bukan** penyerang yang beradaptasi terhadap model, jadi anggap sebagai indikasi, bukan bukti ketahanan.

## Instalasi dan menjalankan

**Prasyarat:** Python 3.10+ dan `pip`. Tidak butuh GPU.

```bash
# 1. Masuk ke folder proyek
cd sms-spam-detector-id

# 2. Buat virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Pasang dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Buka `http://localhost:8501`.

> **Pertama kali dibuka:** model `.joblib` tidak kompatibel antar versi scikit-learn. Jika versi di komputermu berbeda dari versi saat model dilatih, aplikasi menampilkan peringatan dan tombol **"Latih ulang model sekarang"**. Klik sekali (±15 detik) dan aplikasi siap dipakai. Ini mencegah error `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`.

### Melatih ulang model (opsional)

Lakukan jika Anda mengganti dataset, mengubah preprocessing, atau muncul error versi scikit-learn saat memuat model.

```bash
python -m src.train
```

Hasilnya menimpa isi `artifacts/`. Hasil training disimpan beserta versi scikit-learn-nya (`artifacts/versions.json`). Waktu latih sekitar 10–30 detik untuk 2 model (bergantung spesifikasi laptop).

### Menjalankan tes

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

### Mengganti dataset

Ganti `data/indo_spam.csv` dengan berkas yang punya kolom `Kategori` (`ham`/`Spam`) dan `Pesan`, lalu jalankan `python -m src.train`. Untuk format kolom lain, ubah fungsi `load_dataset` di `src/data.py`.

## Opsional: IndoBERT di Google Colab

Untuk membandingkan model klasik dengan *fine-tuning* IndoBERT (seperti pada notebook referensi):

1. Unggah folder proyek ke Colab dan aktifkan GPU (Runtime → Change runtime type → T4).
2. Jalankan:
   ```bash
   pip install -r requirements-indobert.txt
   python -m src.train                  # jika artifacts/ belum ada
   python scripts/train_indobert.py
   ```
3. Skrip memakai split data yang **sama** dengan model klasik, lalu menambahkan IndoBERT ke `artifacts/metrics.json` dan `artifacts/test_predictions.csv`. Unduh folder `artifacts/` yang sudah diperbarui; IndoBERT otomatis muncul di halaman Evaluasi dan Analisis Kesalahan.

Catatan:
- Skrip ini **belum diuji** di lingkungan pengembangan karena butuh GPU dan akses Hugging Face. Jika error karena beda versi `transformers`, sesuaikan argumen `TrainingArguments`.
- Aplikasi web **belum** memuat IndoBERT untuk prediksi langsung (modelnya ±450 MB); IndoBERT hanya muncul di halaman evaluasi.
- Notebook referensi memakai `num_labels=3` padahal datanya biner dan menghitung `classification_report` dengan urutan argumen terbalik (`prediction, actual`), yang menukar precision dan recall. Skrip di sini sudah memperbaiki keduanya (2 kelas, dan `y_true` lebih dulu).

## Keterbatasan

- **Dataset kecil dan tampaknya hasil terjemahan** (lihat catatan di bagian Dataset). Generalisasi ke SMS penipuan Indonesia yang asli belum terbukti.
- **Data uji kecil** (120 pesan Spam); selisih antar model beberapa pesan tidak signifikan.
- **Probabilitas cenderung ekstrem** pada Logistic Regression (C=100); jangan membaca "99,9%" sebagai kepastian.
- **Hanya teks.** Nomor pengirim, tautan yang sebenarnya, dan konteks percakapan tidak dipakai.
- **Pola baru** (modus penipuan yang belum ada di data) bisa lolos; model perlu dilatih ulang secara berkala.
- Uji ketahanan hanya simulasi evasi sederhana.

## Etika dan privasi

- Aplikasi **tidak menyimpan** pesan yang diketik pengguna.
- Dataset berisi teks pesan; jika menambah data dari SMS pribadi, **anonimkan** nomor telepon, nama, dan nomor rekening terlebih dahulu.
- Proyek ini murni **defensif** (mendeteksi ancaman); tidak ada kode atau payload serangan. Fungsi di `src/robustness.py` hanya mensimulasikan penyamaran teks pada data uji untuk mengukur ketahanan model.

## Pemetaan ke luaran tugas

| Ketentuan tugas | Lokasi di proyek |
|---|---|
| Model/aplikasi yang bisa dijalankan | `streamlit run app.py` |
| Evaluasi: precision, recall, F1 | Halaman **Evaluasi Model**; `artifacts/metrics.json` |
| Analisis false positive dan false negative | Halaman **Analisis Kesalahan** |
| Laporan: pendahuluan, metode, pembahasan, kesimpulan | Bahan: bagian Dataset, Metode, Hasil, Keterbatasan di README ini |
| Video demo 5–10 menit | Saran alur: (1) masalah smishing, (2) demo 5 contoh pesan, (3) halaman evaluasi dan perbandingan model, (4) analisis kesalahan dan uji ketahanan, (5) keterbatasan |

## Pemecahan masalah

| Masalah | Solusi |
|---|---|
| `Artefak model belum ada` di web | Jalankan `python -m src.train` dari folder root proyek. |
| `ModuleNotFoundError: No module named 'src'` | Jalankan perintah dari folder root proyek (yang berisi `app.py`), bukan dari dalam `src/`. |
| `AttributeError: ... has no attribute 'multi_class'` atau peringatan versi saat memuat model | Versi scikit-learn berbeda dari saat training. Klik tombol **Latih ulang model sekarang** di aplikasi, atau jalankan `python -m src.train`. |
| `width="stretch"` tidak dikenali | Streamlit terlalu lama. `pip install -U "streamlit>=1.50"`. |
| Port 8501 terpakai | `streamlit run app.py --server.port 8502` |

## Atribusi

- Dataset: *Indonesian SMS Spam Detection* (Kaggle); berkas `indo_spam.csv` dari repositori `gevabriel/dataset`. Lisensi dan ketentuan penggunaan mengikuti halaman sumber, silakan periksa sebelum mendistribusikan ulang.
- Dataset asli yang tampaknya menjadi sumber terjemahannya: Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng.
- Model IndoBERT (opsional): `indobenchmark/indobert-base-p2` (Wilie dkk., 2020, IndoNLU).
- Library: scikit-learn, Streamlit, pandas, matplotlib.
