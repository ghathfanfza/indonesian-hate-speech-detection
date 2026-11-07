# Deteksi Ujaran Kebencian dan Kata Kasar Bahasa Indonesia

Klasifikasi tweet berbahasa Indonesia menjadi **abusive/hate speech** atau **non-abusive**. Dua arsitektur hybrid deep learning dibandingkan, yaitu **CNN + Transformer** dan **LSTM + Transformer**, lalu hasilnya dijelaskan dengan **SHAP**.

> Proyek mini, dikerjakan di Google Colab. Dataset dan sampel di notebook memuat kata-kata kasar.

## Dataset

[Indonesian Abusive and Hate Speech Twitter Text](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text) (Kaggle):

| Berkas | Isi |
|---|---|
| `data.csv` | 13.169 tweet berlabel |
| `abusive.csv` | 125 kata kasar (leksikon) |
| `new_kamusalay.csv` | 15.167 pasangan kata alay ke bentuk formal |

Dataset tidak disertakan di repositori, silakan cek lisensinya di Kaggle.

## Alur

1. **Analisis data.** Distribusi label dan karakteristik teks.
2. **Preprocessing.** Normalisasi kata alay dengan kamus, pembersihan teks, tokenisasi (kosakata 15.000, panjang urutan 40), dan fitur manual `abusive_count` (jumlah kata kasar dari leksikon). Data dibagi 80:20 secara stratified (10.535 latih, 2.634 uji).
3. **Arsitektur hybrid.**
   - *CNN + Transformer:* Embedding (128), Multi-Scale CNN (kernel 2, 3, 4), Transformer encoder, global average pooling, digabung dengan fitur manual, lalu Dense.
   - *LSTM + Transformer:* Embedding, LSTM, Transformer encoder, digabung dengan fitur manual, lalu Dense.
4. **Training dan tuning.** Konfigurasi diuji dengan K-Fold 5 lipatan (30 training run), memakai EarlyStopping, ReduceLROnPlateau, dan ModelCheckpoint.
5. **Validasi.** K-Fold stratified dan evaluasi pada test set.
6. **Interpretasi (XAI).** SHAP pada kalimat contoh.

## Hasil

**Test set (2.634 sampel)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CNN + Transformer | 0,9039 | 0,8913 | 0,8533 | 0,8719 | 0,9589 |
| LSTM + Transformer | **0,9096** | **0,9279** | 0,8285 | **0,8754** | **0,9629** |

**K-Fold 5 lipatan (rata-rata ± simpangan baku)**

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| CNN + Transformer | 0,9009 ± 0,0069 | 0,8670 ± 0,0085 | 0,9592 ± 0,0047 |
| LSTM + Transformer | 0,8829 ± 0,0083 | 0,8430 ± 0,0080 | 0,9446 ± 0,0079 |

Pada test set, LSTM + Transformer unggul tipis, sedangkan pada validasi silang CNN + Transformer lebih tinggi. Selisih keduanya kecil, jadi kedua model layak dianggap sebanding. Kedua model menunjukkan tingkat memorization yang rendah.

Kelas "abusive" pada LSTM + Transformer: precision 0,93, recall 0,83, F1 0,88 (kelas "non-abusive": F1 0,93).

**Interpretasi (SHAP).** Kata berwarna merah mendorong prediksi ke *abusive*, kata berwarna biru mendorong ke *non-abusive*. Pada sampel uji, kalimat dengan makian mendapat probabilitas abusive tinggi, sedangkan kalimat netral seperti ucapan terima kasih atau pertanyaan informasi mendapat probabilitas rendah.

[Tambahkan grafik hasil (`training_comparison.png`, `confusion_matrices.png`, `validasi_laporan_lengkap.png`) ke folder `images/`.]

## Keterbatasan

Model bisa salah pada sarkasme, konteks, dan slang baru, serta dilatih pada data Twitter dari periode tertentu. Gunakan hanya sebagai alat bantu, bukan pengganti peninjauan manusia.

## Menjalankan

Cara termudah: unggah `notebooks/hate_speech_detection.ipynb` ke Google Colab dan jalankan semua sel.
Secara lokal:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/hate_speech_detection.ipynb
```
Dataset diunduh otomatis lewat `kagglehub` dan dapat meminta login Kaggle.

## Struktur

```
notebooks/   notebook analisis, training, validasi, dan interpretasi
requirements.txt
```

## Teknologi

Python, TensorFlow/Keras, scikit-learn, SHAP, pandas, matplotlib.

## Peran

Peran saya: **Programmer dan Project Manager**. [Tambahkan anggota tim jika ada, setelah mendapat izin.]

## Lisensi

[Pilih lisensi, mis. MIT, lalu tambahkan file LICENSE]
