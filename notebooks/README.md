# 📁 /notebooks — Core Project Logic

All key analysis is implemented in well-separated notebooks. Each corresponds to a **specific phase** of the predictive maintenance pipeline.

---

## 🧪 Notebooks

### 🔹 `01_data_cleaning.ipynb` — Phase 1
- Reads all raw `.txt` files
- Applies:
  - Column naming
  - Constant sensor detection & removal
  - Min-max normalization
- Saves cleaned files to `/data/clean_train_*.csv`

---

### 🔹 `02_clustering.ipynb` — Phase 2
- Loads cleaned data
- Runs both:
  - KMeans (k=5)
  - AgglomerativeClustering (n=5)
- Computes silhouette scores
- Applies dimensionality reduction:
  - PCA
  - t-SNE
- Exports:
  - Clustered CSVs (with stage labels)
  - All visualizations to `/figures/`
  - Evaluation metrics to `/report/`
