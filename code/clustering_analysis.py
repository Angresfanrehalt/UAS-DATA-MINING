"""
=============================================================
Tugas Proyek Akhir: Preprocessing & Clustering
Dataset: Mall Customer Segmentation
Nama: Angres Fanrehalt
Nim: 241011400778
Kelas: 04TPLM008
Mata Kuliah: DATA MINING
=============================================================
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 60)
print("PROYEK DATA MINING: PREPROCESSING & CLUSTERING")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("\n[1] LOAD DATASET")
df_raw = pd.read_csv("/workspaces/UAS-DATA-MINING/data/mall_customers_raw.csv")
print(f"Shape awal  : {df_raw.shape}")
print(f"Kolom       : {list(df_raw.columns)}")
print("\nSample data (5 baris pertama):")
print(df_raw.head())
print("\nInfo dataset:")
print(df_raw.info())

# ─────────────────────────────────────────────
# 2. CLEANING DATA
# ─────────────────────────────────────────────
print("\n[2] CLEANING DATA")
df = df_raw.copy()

# 2a. Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Annual_Income_k"].fillna(df["Annual_Income_k"].median(), inplace=True)
df["Spending_Score"].fillna(df["Spending_Score"].median(), inplace=True)
print(f"Missing values setelah imputasi: {df.isnull().sum().sum()}")

# 2b. Duplikat
print("\n--- Duplikat ---")
print(f"Duplikat ditemukan: {df.duplicated().sum()} baris")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Shape setelah hapus duplikat: {df.shape}")

# 2c. Standarisasi Gender
print("\n--- Standardisasi Kolom Kategorikal ---")
print(f"Nilai unik Gender sebelum: {df['Gender'].unique()}")
df["Gender"] = df["Gender"].str.lower().str.strip()
df["Gender"] = df["Gender"].replace({"m": "male", "f": "female"})
print(f"Nilai unik Gender sesudah: {df['Gender'].unique()}")

# 2d. Encoding Gender
df["Gender_Enc"] = df["Gender"].map({"male": 0, "female": 1})

# 2e. Simpan cleaned dataset
df.to_csv("/workspaces/UAS-DATA-MINING/data/mall_customers_cleaned.csv", index=False)
print(f"\nDataset bersih disimpan: {df.shape}")

# ─────────────────────────────────────────────
# 3. SCALING / NORMALISASI
# ─────────────────────────────────────────────
print("\n[3] SCALING DATA")
features = ["Age", "Annual_Income_k", "Spending_Score"]
X = df[features].copy()

scaler = StandardScaler()
X_clean = X.dropna()
X_scaled = scaler.fit_transform(X_clean)
df_clean_idx = X_clean.index
df = df.loc[df_clean_idx].reset_index(drop=True)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("Statistik setelah StandardScaler:")
print(X_scaled_df.describe().round(3))

# ─────────────────────────────────────────────
# 4. MENENTUKAN K OPTIMAL
# ─────────────────────────────────────────────
print("\n[4] MENENTUKAN K OPTIMAL (Elbow + Silhouette)")
inertias, silhouettes = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(
        f"  k={k} | Inertia={km.inertia_:.1f} | Silhouette={silhouette_score(X_scaled, labels):.4f}"
    )

best_k = k_range[np.argmax(silhouettes)]
print(f"\n>> K terbaik berdasarkan Silhouette Score: {best_k}")

# ─────────────────────────────────────────────
# 5. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print(f"\n[5] K-MEANS CLUSTERING (k={best_k})")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score final: {final_sil:.4f}")

print("\nDistribusi cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nKarakteristik tiap cluster:")
cluster_profile = (
    df.groupby("Cluster")[features + ["Gender"]]
    .agg(
        {
            "Age": "mean",
            "Annual_Income_k": "mean",
            "Spending_Score": "mean",
            "Gender": lambda x: x.mode()[0],
        }
    )
    .round(2)
)
print(cluster_profile)

# ─────────────────────────────────────────────
# 6. VISUALISASI
# ─────────────────────────────────────────────
print("\n[6] MEMBUAT VISUALISASI...")

palette = sns.color_palette("Set2", best_k)
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Hasil Clustering Mall Customers", fontsize=16, fontweight="bold", y=0.98)

# Plot 1: Elbow Method
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=6)
ax1.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"k={best_k}")
ax1.set_title("Elbow Method", fontweight="bold")
ax1.set_xlabel("Jumlah Cluster (k)")
ax1.set_ylabel("Inertia")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Silhouette Score
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(list(k_range), silhouettes, "rs-", linewidth=2, markersize=6)
ax2.axvline(best_k, color="blue", linestyle="--", alpha=0.7, label=f"k={best_k}")
ax2.set_title("Silhouette Score", fontweight="bold")
ax2.set_xlabel("Jumlah Cluster (k)")
ax2.set_ylabel("Silhouette Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Income vs Spending Score
ax3 = fig.add_subplot(3, 3, 3)
for i in range(best_k):
    mask = df["Cluster"] == i
    ax3.scatter(
        df.loc[mask, "Annual_Income_k"],
        df.loc[mask, "Spending_Score"],
        c=[palette[i]],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
ax3.set_title("Income vs Spending Score", fontweight="bold")
ax3.set_xlabel("Annual Income (k$)")
ax3.set_ylabel("Spending Score")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Age vs Spending Score
ax4 = fig.add_subplot(3, 3, 4)
for i in range(best_k):
    mask = df["Cluster"] == i
    ax4.scatter(
        df.loc[mask, "Age"],
        df.loc[mask, "Spending_Score"],
        c=[palette[i]],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
ax4.set_title("Age vs Spending Score", fontweight="bold")
ax4.set_xlabel("Age")
ax4.set_ylabel("Spending Score")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Age vs Income
ax5 = fig.add_subplot(3, 3, 5)
for i in range(best_k):
    mask = df["Cluster"] == i
    ax5.scatter(
        df.loc[mask, "Age"],
        df.loc[mask, "Annual_Income_k"],
        c=[palette[i]],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
ax5.set_title("Age vs Annual Income", fontweight="bold")
ax5.set_xlabel("Age")
ax5.set_ylabel("Annual Income (k$)")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
ax6 = fig.add_subplot(3, 3, 6)
for i in range(best_k):
    mask = df["Cluster"] == i
    ax6.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=[palette[i]],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
centers_pca = pca.transform(kmeans.cluster_centers_)
ax6.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c="black",
    marker="X",
    s=200,
    zorder=5,
    label="Centroid",
)
ax6.set_title(
    f"PCA 2D (var={sum(pca.explained_variance_ratio_) * 100:.1f}%)", fontweight="bold"
)
ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7: Cluster Distribution Pie
ax7 = fig.add_subplot(3, 3, 7)
counts = df["Cluster"].value_counts().sort_index()
ax7.pie(
    counts,
    labels=[f"Cluster {i}\n({v} orang)" for i, v in counts.items()],
    colors=palette,
    autopct="%1.1f%%",
    startangle=90,
)
ax7.set_title("Distribusi Cluster", fontweight="bold")

# Plot 8: Boxplot Spending Score per Cluster
ax8 = fig.add_subplot(3, 3, 8)
data_box = [df[df["Cluster"] == i]["Spending_Score"].values for i in range(best_k)]
bp = ax8.boxplot(data_box, patch_artist=True)
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
ax8.set_title("Spending Score per Cluster", fontweight="bold")
ax8.set_xlabel("Cluster")
ax8.set_ylabel("Spending Score")
ax8.set_xticklabels([f"C{i}" for i in range(best_k)])
ax8.grid(True, alpha=0.3, axis="y")

# Plot 9: Radar/Bar profile
ax9 = fig.add_subplot(3, 3, 9)
profile_norm = cluster_profile[features].copy()
for col in features:
    profile_norm[col] = (profile_norm[col] - profile_norm[col].min()) / (
        profile_norm[col].max() - profile_norm[col].min()
    )
x = np.arange(len(features))
width = 0.8 / best_k
for i in range(best_k):
    ax9.bar(
        x + i * width,
        profile_norm.iloc[i],
        width,
        label=f"Cluster {i}",
        color=palette[i],
        alpha=0.8,
    )
ax9.set_title("Profil Cluster (Normalized)", fontweight="bold")
ax9.set_xticks(x + width * (best_k - 1) / 2)
ax9.set_xticklabels(["Age", "Income", "Spending"], fontsize=9)
ax9.set_ylabel("Nilai Ternormalisasi")
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("visualisasi_clustering.png", dpi=150, bbox_inches="tight")
print("Visualisasi disimpan: visualisasi_clustering.png")

# ─────────────────────────────────────────────
# 7. SIMPAN HASIL
# ─────────────────────────────────────────────
df.to_csv("/workspaces/UAS-DATA-MINING/data/mall_customers_with_clusters.csv", index=False)
print("\nDataset dengan label cluster disimpan: /workspaces/UAS-DATA-MINING/data/mall_customers_with_clusters.csv")

print("\n" + "=" * 60)
print("ANALISIS SELESAI")
print(f"Silhouette Score: {final_sil:.4f}")
print(f"Jumlah Cluster Optimal: {best_k}")
print("=" * 60)
