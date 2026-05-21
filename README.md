# Computer Price Prediction

Proyek akhir mata kuliah **Dasar Analitik Data (DAD)** yang membangun sistem prediksi harga komputer kustom menggunakan machine learning dengan mengikuti metodologi CRISP-DM secara menyeluruh.

- **Penyusun:** Marshal Aufa Diliyana (NPM: 2406346913)
- **Kelas:** Dasar Analitik-08
- **Dosen:** Dr. Taufiq Alif Kurniawan
- **Dataset:** `computer_prices_all.csv` -- 100.000 baris x 33 kolom

---

## Video Penjelasan

[![Video Penjelasan Proyek](https://img.youtube.com/vi/XeDfnJrAhvM/maxresdefault.jpg)](https://youtu.be/XeDfnJrAhvM?si=dbXC1E7flBX991nE)

Klik gambar di atas atau buka langsung: https://youtu.be/XeDfnJrAhvM?si=dbXC1E7flBX991nE

---

## Deskripsi Proyek

Menentukan harga jual komputer kustom secara manual memerlukan waktu yang lama dan menghasilkan harga yang tidak konsisten. Proyek ini membangun model prediksi berbasis supervised machine learning yang dapat menentukan estimasi harga wajar secara instan berdasarkan 33 parameter spesifikasi hardware.

**Target analitis:** Memprediksi variabel kontinu `price` dengan R-squared di atas 85%.

---

## Struktur Repositori

```
.
├── pipeline.py                  # Skrip pipeline utama end-to-end CRISP-DM
├── computer_prices_all.csv      # Dataset mentah (100.000 baris x 33 kolom)
├── laporan-proyek-akhir.pdf     # Laporan PDF final project
├── .gitignore
├── README.md
├── figures/                     # 20 grafik visualisasi hasil pipeline
│   ├── image_01_price_distribution.png
│   ├── image_02_categorical_distributions.png
│   ├── image_03_correlation_heatmap.png
│   ├── image_04_brand_vs_price_boxplot.png
│   ├── image_05_ram_vs_price_scatter.png
│   ├── image_06_weight_vs_price_density.png
│   ├── image_07_outliers_before_after.png
│   ├── image_08_pivot_average_price.png
│   ├── image_09_pivot_distribution_count.png
│   ├── image_10_binned_price_distribution.png
│   ├── image_11_full_encoded_correlation.png
│   ├── image_12_random_forest_importances.png
│   ├── image_13_feature_selection_comparison.png
│   ├── image_14_rfe_ridge_coefficients.png
│   ├── image_15_model_performance_comparison.png
│   ├── image_16_model_mae_comparison.png
│   ├── image_17_best_model_actual_vs_predicted.png
│   ├── image_18_best_model_residuals.png
│   ├── image_19_decision_tree_vs_rf.png
│   └── image_20_model_learning_generalization.png
├── data/                        # Dataset antara dan ringkasan statistik
│   ├── clean_df.csv
│   ├── df_filtered.csv
│   ├── df_wrapper.csv
│   ├── df_embedded.csv
│   ├── comprehensive_numerical_statistics.csv
│   ├── feature_selection_comparison.csv
│   ├── model_performance_comparison.csv
│   ├── q1_bottom_10_rows.csv
│   └── q4_specific_describe.csv
└── reports/                     # Laporan teks per fase analisis
    ├── handson_data_understanding_report.txt
    ├── handson_data_preparation_report.txt
    ├── handson_data_preparation_3_report.txt
    └── handson_modelling_report.txt
```

---

## Metodologi CRISP-DM

Pipeline mengikuti enam fase standar CRISP-DM yang diimplementasikan secara berurutan dalam satu skrip.

### Phase 1 -- Business and Data Understanding (Topik 01-04)

- Definisi masalah bisnis dan sasaran analitis
- Profiling dataset: tipe data, kardinalitas, missing values audit
- Statistik deskriptif: mean, median, std dev, IQR, skewness, kurtosis
- Visualisasi distribusi univariat dan korelasi bivariat (image 01 hingga 06)

### Phase 2 -- Data Preparation (Topik 05-06)

- Deteksi dan penanganan outlier menggunakan metode IQR Winsorization (clamping)
- Pembuatan pivot table agregasi harga dan volume stok
- Feature binning: harga kontinu dikategorikan menjadi Low, Medium, High
- Feature encoding: One-Hot Encoding (nominal) dan Ordinal Encoding (ordinal)
- Visualisasi sebelum-sesudah outlier treatment (image 07 hingga 10)

### Phase 3 -- Feature Selection (Topik 07)

Tiga metode seleksi fitur diuji dan dibandingkan:

| Metode | Pendekatan | Fitur Terpilih | R2 (KNN) | MAE |
|---|---|---|---|---|
| Filter Method | Korelasi Pearson + eliminasi multikolinearitas | 16 | 0.8482 | $173.27 |
| Wrapper Method | RFE dengan estimator Ridge | **15** | **0.8556** | **$168.73** |
| Embedded Method | Random Forest Feature Importance | ~18 | 0.8460 | $174.43 |

**Pemenang: Wrapper Method** -- dipilih sebagai dataset input untuk fase pemodelan.

### Phase 4 -- Modelling and Evaluation (Topik 09-12)

Lima algoritma regresi dilatih dan dibandingkan:

| Model | Train R2 | Test R2 | Test MAE | Test RMSE |
|---|---|---|---|---|
| K-Nearest Neighbors (KNN) | -- | -- | -- | -- |
| Multivariable Linear Regression | -- | -- | -- | -- |
| Support Vector Regressor (LinearSVR) | -- | -- | -- | -- |
| Decision Tree Regressor | -- | -- | -- | -- |
| **Random Forest Regressor** | -- | **0.8785** | **$157.13** | -- |

Metrik yang digunakan: R-squared (R2), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).

Optimasi hyperparameter KNN dilakukan dengan GridSearchCV (5-Fold Cross Validation).

**Model terbaik: Random Forest Regressor** -- akurasi pengujian 87.85%, rata-rata kesalahan tebakan $157.13.

---

## Cara Menjalankan

### Prasyarat

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Menjalankan Pipeline

```bash
python pipeline.py
```

Pipeline akan otomatis membuat folder `figures/`, `data/`, dan `reports/` lalu menghasilkan seluruh artefak secara berurutan.

Perkiraan waktu eksekusi: 5-15 menit tergantung spesifikasi mesin.

---

## Hasil Utama

- Model Random Forest mencapai R2 = 0.8785 pada data testing
- Gap overfitting (Train R2 - Test R2) sangat kecil, membuktikan model tidak overfit
- Residual tersebar acak di sekitar nol (homoskedastisitas terpenuhi)
- Seleksi fitur Wrapper Method berhasil memangkas dari 51 fitur menjadi 15 fitur optimal tanpa penurunan akurasi signifikan

---

## Dependensi

| Library | Versi Minimum | Kegunaan |
|---|---|---|
| pandas | 1.5 | Manipulasi dan pembersihan data |
| numpy | 1.23 | Komputasi numerik |
| scikit-learn | 1.2 | Pemodelan dan evaluasi ML |
| matplotlib | 3.6 | Visualisasi dasar |
| seaborn | 0.12 | Visualisasi statistik |
| scipy | 1.9 | Kalkulasi skewness dan kurtosis |