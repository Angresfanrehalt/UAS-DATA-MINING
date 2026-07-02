# 📁 Struktur Folder Tugas Clustering

```
tugas-clustering-online-retail/
├── code/
│   └── clustering_analysis.py
├── data/
│   └── mall_customer_cleaned.csv
│   └── mall_customer_raw.csv
│   └── mall_customer_with_cluster.csv          
├── notebook/
│   └── clustering_notebook.ipynb   
├── laporan/
│   └── laporan.md        
├── outputs/                          
├── requirements.txt                  
├── README.md
├── visualisasi_clustering.png        
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
