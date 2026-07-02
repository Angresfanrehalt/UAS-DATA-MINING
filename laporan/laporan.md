# Laporan Proyek Akhir Data Mining
## Preprocessing dan Unsupervised Learning (Clustering)

**Nama Dataset**: Mall Customer Segmentation (Synthetic)  
**Algoritma**: K-Means Clustering  
**Bahasa Pemrograman**: Python 3 (Scikit-Learn, Pandas, Matplotlib, Seaborn)

---

## 1. Pendahuluan

Proyek ini bertujuan untuk mengelompokkan pelanggan mall berdasarkan karakteristik demografis dan perilaku belanja mereka. Dataset yang digunakan merupakan data sintetis berbasis konsep *Mall Customer Segmentation* yang umum digunakan pada kasus clustering.

**Tujuan utama:**
- Melakukan pembersihan data mentah yang mengandung cacat (missing values, duplikat, format tidak konsisten)
- Mengimplementasikan algoritma K-Means Clustering
- Menemukan segmen pelanggan yang bermakna sebagai insight bisnis

---

## 2. Deskripsi Dataset

| Atribut | Keterangan |
|---|---|
| **CustomerID** | ID unik setiap pelanggan |
| **Gender** | Jenis kelamin (Male/Female) |
| **Age** | Usia pelanggan (tahun) |
| **Annual_Income_k** | Pendapatan tahunan (dalam ribuan USD) |
| **Spending_Score** | Skor pengeluaran 1–100 (diberikan mall) |

**Ukuran dataset awal**: 220 baris × 5 kolom

**Kecacatan data yang ditemukan:**
- Missing values: 33 nilai kosong (Age: 13, Annual_Income_k: 11, Spending_Score: 9)
- Duplikat: 10 baris duplikat
- Format tidak konsisten: kolom Gender memiliki 6 format berbeda (`Male`, `Female`, `male`, `female`, `M`, `F`)

---

## 3. Tahap Cleaning Data

### 3.1 Penanganan Missing Values
- Kolom numerik (`Age`, `Annual_Income_k`, `Spending_Score`) diimputasi menggunakan **nilai median** karena lebih robust terhadap outlier dibanding mean.

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Annual_Income_k'].fillna(df['Annual_Income_k'].median(), inplace=True)
df['Spending_Score'].fillna(df['Spending_Score'].median(), inplace=True)
```

### 3.2 Penanganan Duplikat
- Ditemukan **10 baris duplikat** yang dihapus menggunakan `drop_duplicates()`.
- Ukuran dataset setelah penghapusan: **210 baris**.

### 3.3 Standarisasi Data Kategorikal (Gender)
- Nilai Gender dinormalisasi ke huruf kecil, lalu `M` → `male` dan `F` → `female`.
- Setelah standarisasi: hanya ada 2 nilai unik (`male`, `female`).
- Dilakukan **Label Encoding**: `male` = 0, `female` = 1.

### 3.4 Scaling / Normalisasi
- Fitur numerik (`Age`, `Annual_Income_k`, `Spending_Score`) distandarisasi menggunakan **StandardScaler** (z-score normalization).
- Hasil: mean ≈ 0, std ≈ 1 untuk setiap fitur.

---

## 4. Implementasi Clustering (K-Means)

### 4.1 Penentuan Nilai K Optimal

Dua metode digunakan secara bersamaan:

**Elbow Method** — mencari titik "siku" pada kurva inertia.  
**Silhouette Score** — mengukur seberapa baik setiap sampel cocok di clusternya (semakin mendekati 1 = semakin baik).

| k | Inertia | Silhouette Score |
|---|---|---|
| 2 | 403.9 | 0.2473 |
| 3 | 314.1 | 0.2537 |
| 4 | 235.7 | 0.3042 |
| 5 | 197.6 | 0.2932 |
| 6 | 163.2 | 0.3201 |
| **7** | **138.9** | **0.3294** ✓ |
| 8 | 126.2 | 0.3137 |

**Kesimpulan**: K = 7 dipilih karena menghasilkan Silhouette Score tertinggi (0.3294).

### 4.2 Hasil K-Means

**Silhouette Score Final: 0.3294**

---

## 5. Profil dan Interpretasi Cluster

| Cluster | Usia Rata-rata | Income (k$) | Spending Score | Gender Dominan | Interpretasi |
|---|---|---|---|---|---|
| 0 | 53 | 122 | 28 | Male | Pelanggan kaya, tua, belanja rendah |
| 1 | 34 | 46 | 76 | Male | Pelanggan muda, income menengah, suka belanja |
| 2 | 23 | 29 | 20 | Female | Pelanggan muda, income rendah, jarang belanja |
| 3 | 36 | 113 | 81 | Male | Pelanggan kaya, aktif belanja — **Target Utama** |
| 4 | 55 | 51 | 28 | Female | Pelanggan setengah baya, income sedang, belanja rendah |
| 5 | 61 | 80 | 76 | Female | Pelanggan senior, income menengah, suka belanja |
| 6 | 29 | 98 | 26 | Male | Pelanggan muda kaya, belanja rendah — **Potensi Tinggi** |

---

## 6. Visualisasi

File visualisasi disimpan sebagai `visualisasi_clustering.png`, terdiri dari:
1. **Elbow Method** — kurva inertia vs jumlah cluster
2. **Silhouette Score** — kurva skor vs jumlah cluster
3. **Scatter Plot: Income vs Spending Score** per cluster
4. **Scatter Plot: Age vs Spending Score** per cluster
5. **Scatter Plot: Age vs Income** per cluster
6. **PCA 2D** — reduksi dimensi untuk visualisasi clustering
7. **Pie Chart** — distribusi jumlah anggota tiap cluster
8. **Boxplot Spending Score** per cluster
9. **Bar Chart Profil Cluster** (ternormalisasi)

---

## 7. Kesimpulan

1. **Preprocessing berhasil** membersihkan 33 missing values, 10 duplikat, dan 6 format Gender berbeda menjadi dataset bersih 210 baris.
2. **K-Means dengan k=7** menghasilkan Silhouette Score 0.3294, menunjukkan pemisahan cluster yang cukup baik.
3. **Cluster paling valuable** secara bisnis adalah **Cluster 3** (income tinggi + spending score tinggi) dan **Cluster 6** (income tinggi, spending score rendah = potensi untuk ditingkatkan).
4. **Rekomendasi strategi**: Mall dapat memberikan program loyalitas khusus untuk Cluster 3, dan promosi terarah untuk mengaktivasi Cluster 6.

---

## 8. File yang Dikumpulkan

| File | Keterangan |
|---|---|
| `mall_customers_raw.csv` | Dataset mentah (dengan missing values & duplikat) |
| `mall_customers_cleaned.csv` | Dataset setelah cleaning |
| `mall_customers_with_clusters.csv` | Dataset dengan label cluster hasil K-Means |
| `clustering_analysis.py` | Source code analisis lengkap |
| `clustering_notebook.ipynb` | Notebook Jupyter dengan penjelasan step-by-step |
| `visualisasi_clustering.png` | Output visualisasi 9 panel |
| `laporan.md` | Laporan ini |

---

*Proyek Akhir Data Mining | Unsupervised Learning (Clustering)*
