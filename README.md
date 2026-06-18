# GamU - Machine Learning Repository

Repositori ini memuat *source code* dan *pipeline* yang digunakan untuk memproses data dan melatih model Machine Learning pada sistem rekomendasi GamU.

## Daftar Isi
- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Dataset](#dataset)
- [Hasil dan Pengujian](#hasil-dan-pengujian)
- [Persyaratan (Prerequisites)](#persyaratan-prerequisites)
- [Instalasi dan Penggunaan](#instalasi-dan-penggunaan)
- [Struktur Direktori](#struktur-direktori)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

## Tentang Proyek
Sistem rekomendasi GamU dibangun menggunakan pendekatan **Content-Based Filtering**. Berbeda dengan *Collaborative Filtering*, model ini tidak mengandalkan riwayat interaksi pengguna lain, sehingga terhindar dari *Cold Start Problem*. Model bekerja dengan menganalisis fitur teks (seperti *tags* dan deskripsi cerita) dari setiap *game* untuk menemukan tingkat kemiripan antar *game*.

## Fitur Utama
* **Pembersihan Data Otomatis:** Skrip memfilter data yang tidak relevan atau memiliki atribut teks yang kosong.
* **Ekstraksi Fitur Teks:** Mengimplementasikan `TfidfVectorizer` untuk mengubah teks natural menjadi representasi bobot numerik.
* **Pengukuran Kemiripan Akurat:** Menggunakan `cosine_similarity` untuk menghitung jarak vektor antar entitas *game* dengan presisi tinggi.
* **Export Model Siap Pakai:** Menyimpan matriks dan model ke format `.pkl` agar dapat langsung digunakan oleh *backend* atau *embedded system*.

## Dataset
Model ini dilatih menggunakan dataset metadata *game* dari Steam. Pada tahap pra-pemrosesan, sistem melakukan penyaringan terhadap data yang tidak memiliki atribut *tags*, sehingga menghasilkan **40.484 data game bersih** yang siap diproses sebagai basis rekomendasi. Proses TF-IDF pada dataset ini menghasilkan 471 fitur unik (*tags*).

## Hasil dan Pengujian
Pengujian skenario rekomendasi dilakukan menggunakan skrip di dalam *notebook*. Sebagai contoh, saat sistem diminta mencari rekomendasi untuk entri **"METAL SLUG 3"**, model berhasil memberikan hasil teratas yang sangat relevan:
1. **METAL SLUG X** (Skor Kemiripan: 0.852)
2. **METAL SLUG 2** (Skor Kemiripan: 0.765)
3. **METAL SLUG** (Skor Kemiripan: 0.746)
4. **Contra Anniversary Collection** (Skor Kemiripan: 0.734)
5. **Super Cyborg** (Skor Kemiripan: 0.718)

## Persyaratan (Prerequisites)
Pastikan sistem Anda telah terinstal Python 3.x beserta beberapa pustaka pendukung berikut:
* `pandas`
* `scikit-learn`
* `jupyter`

## Instalasi dan Penggunaan
1. Lakukan *clone* pada repositori ini ke dalam mesin lokal Anda:
    ```bash
    git clone [https://github.com/USERNAME_ORGANIZATION/gamu-ml.git](https://github.com/USERNAME_ORGANIZATION/gamu-ml.git)
    cd gamu-ml
    ```

2. Instal dependensi yang dibutuhkan melalui *pip*:
    ```bash
    pip install pandas scikit-learn jupyter
    ```

3. Buka *notebook* eksperimen utama:
    ```bash
    jupyter notebook notebooks/02_recommendation_model.ipynb
    ```

4. Jalankan seluruh sel kode (*Run All*) di dalam *notebook* tersebut. Skrip akan secara otomatis memproses data, melatih model, dan menghasilkan berkas keluaran model.

5. Berkas model yang berhasil dibuat (`tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`, dan `clean_games_df.pkl`) akan tersimpan secara otomatis di dalam direktori `models/`.

## Struktur Direktori

    gamu-ml/
    ├── data/
    │   └── processed/
    │       └── clean_data_games.csv
    ├── models/
    │   ├── tfidf_vectorizer.pkl
    │   ├── tfidf_matrix.pkl
    │   └── clean_games_df.pkl
    ├── notebooks/
    │   └── 02_recommendation_model.ipynb
    └── README.md

## Kontribusi
Jika Anda ingin berkontribusi pada repositori ini, silakan buat *Pull Request* atau laporkan masalah (*issues*) pada tab yang tersedia di GitHub. Pastikan kode Anda mengikuti standar penulisan Python (PEP 8).

## Lisensi
Proyek ini didistribusikan di bawah lisensi MIT. Lihat berkas `LICENSE` untuk informasi lebih lanjut.
