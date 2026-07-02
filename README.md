# 📁 Struktur Folder Tugas Clustering

```
tugas_clustering/
├── mall_customers_raw.csv         
├── mall_customers_cleaned.csv     
├── mall_customers_with_clusters.csv
├── clustering_analysis.py          
├── clustering_notebook.ipynb       
├── visualisasi_clustering.png      
├── laporan.md                      
└── README.md                       
```

## 🚀 Cara Menjalankan

### Opsi 1: Jalankan script Python
```bash
python clustering_analysis.py
```

### Opsi 2: Buka Jupyter Notebook
```bash
jupyter notebook clustering_notebook.ipynb
```

## 📋 Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
Install: `pip install pandas numpy scikit-learn matplotlib seaborn`

## 📊 Hasil
- Dataset: Mall Customer Segmentation
- Algoritma: K-Means Clustering
- K Optimal: 7 (Silhouette Score: 0.3294)
