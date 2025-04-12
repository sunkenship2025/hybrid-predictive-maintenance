# 📁 /figures — Clustering and Sensor Behavior Visualizations

This folder contains **all visual plots** generated during **Phase 2** of the Hybrid Predictive Maintenance pipeline, as well as enhancement analyses. These plots are essential for:

- 🔍 Interpreting unsupervised clustering behavior
- 📉 Understanding sensor patterns across engine life stages
- 🧠 Communicating insights in the final report

---

## 📚 Plot Categories and Purpose

### 1. 📈 `*_PCA_Clusters.png`
- **What it shows:** KMeans and Agglomerative clusters projected into 2D space using PCA (Principal Component Analysis).
- **Why it's useful:** Helps us understand how clearly the degradation stages separate linearly in reduced dimensions.
- **Generated using:** `sklearn.decomposition.PCA`

➡️ Example:  
`FD002_PCA_Clusters.png`

---

### 2. 🧬 `*_TSNE_Clusters.png`
- **What it shows:** The same clusters as PCA, but using t-SNE (t-Distributed Stochastic Neighbor Embedding).
- **Why it's useful:** Captures non-linear separations and reveals local structure in high-dimensional sensor data.
- **Generated using:** `sklearn.manifold.TSNE`

➡️ Example:  
`FD004_TSNE_Clusters.png`

---

### 3. 📊 `*_Cluster_Distribution.png`
- **What it shows:** Bar chart of how many samples fall into each degradation stage (0 to 4), per clustering method.
- **Why it's useful:** Lets us analyze how balanced the clusters are — e.g., if one stage dominates, modeling may be biased.
- **Used in:** Feature analysis, risk modeling

➡️ Example:  
`FD001_Cluster_Distribution.png`

---

### 4. 🕒 `*_Stage_TimeProfile.png`
- **What it shows:** Average number of cycles (`time`) per KMeans stage.
- **Why it's useful:** Reveals progression of engine health over time. Later stages typically have lower time-to-failure.
- **Great for:** Connecting clusters to Remaining Useful Life (RUL)

➡️ Example:  
`FD003_Stage_TimeProfile.png`

---

### 5. 📉 `*_ElbowCurve.png`
- **What it shows:** KMeans inertia (WSS) vs. number of clusters (k), showing where the curve “elbows”.
- **Why it's useful:** Helps determine the optimal number of clusters (usually 4–6).
- **Used for:** Clustering model validation

➡️ Example:  
`FD002_ElbowCurve.png`

---

### 6. 🌿 `*_Dendrogram.png`
- **What it shows:** Hierarchical Agglomerative clustering tree — shows which samples cluster together and how.
- **Why it's useful:** Explains the structure and linkage between engine instances based on similarity.
- **Generated using:** `scipy.cluster.hierarchy.dendrogram`

➡️ Example:  
`FD004_Dendrogram.png`

---

### 7. 🧪 `*_TopSensors_PerStage.png`
- **What it shows:** Line plot of the **top 3 high-variance sensors** per dataset, averaged over each KMeans stage.
- **Why it's useful:** Shows which sensitive sensors vary across degradation stages and are most relevant for modeling.
- **Used in:** Feature selection for Phase 3 and sensor-based diagnostics.

➡️ Example:  
`FD001_TopSensors_PerStage.png`

---

## 🗂️ File Naming Convention

Each file is named as:

| Part | Meaning |
|------|---------|
| `FD001` | Dataset ID (FD001 to FD004) |
| `PCA_Clusters` | PCA projection of clusters |
| `TSNE_Clusters` | t-SNE projection |
| `Cluster_Distribution` | Cluster counts |
| `Stage_TimeProfile` | Avg time per stage |
| `ElbowCurve` | Elbow method for KMeans |
| `Dendrogram` | Hierarchical tree from Agglo |
| `TopSensors_PerStage` | Top 3 sensors plotted across stages |

---

## 🧠 Final Notes

- ✅ These plots are ready to be embedded in your report or presentation.
- 📊 They support clustering validity, visual storytelling, and degradation interpretation.
- 🏆 High visual quality = high score in competitions and reviews.