# /notebooks — Phase-Wise Implementation Notebooks

This folder contains the core logic and implementation of the Hybrid Predictive Maintenance project. Each notebook corresponds to a specific phase in the pipeline and documents the methodology, code, and outputs associated with that phase.

---

## Notebook Overview

### 1. `01_data_cleaning.ipynb` — Phase 1: Data Preprocessing
This notebook performs preprocessing of the raw NASA CMAPSS datasets (`train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt`).

Key operations include:
- Assigning meaningful column names (e.g., `sensor_1`, `sensor_2`, ...).
- Removing constant-value sensors to eliminate redundancy.
- Normalizing sensor values using Min-Max scaling.
- Saving the cleaned data to the `/data/` folder as:
  - `clean_train_FD001.csv`
  - `clean_train_FD002.csv`
  - `clean_train_FD003.csv`
  - `clean_train_FD004.csv`

These cleaned files are used in clustering and modeling steps in subsequent phases.

---

### 2. `02_clustering.ipynb` — Phase 2: Clustering and Stage Assignment
This notebook focuses on unsupervised clustering to identify degradation stages.

Major tasks include:
- Loading cleaned training datasets from `/data/`.
- Applying KMeans clustering and Agglomerative Clustering (both with 5 clusters).
- Computing silhouette scores to evaluate clustering performance.
- Performing dimensionality reduction using:
  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Saving outputs to:
  - `/data/clustered_train_FD00x.csv` (includes cluster and stage labels)
  - `/figures/` (PCA, t-SNE plots, elbow curves, dendrograms, and time profile visualizations)
  - `/report/` (summary files and evaluation metrics)

---

### 3. `02.5_cluster_verification.ipynb` — Phase 2.5: Manual Cluster Verification and Correction
This notebook performs manual verification and correction of the clustering stages created in Phase 2.

Key activities include:
- Plotting the top 5 high-variance sensors for each FD00x dataset to visually inspect stage progression.
- Detecting coinciding stages using sensor-wise mean comparison.
- Applying manual relabeling wherever necessary to correct overlapping or misaligned cluster stages.
- Saving corrected datasets as:
  - `corrected_clustered_train_FD001.csv`
  - `corrected_clustered_train_FD002.csv`
  - `corrected_clustered_train_FD003.csv`
  - `corrected_clustered_train_FD004.csv`

This phase ensures that health stage labeling is accurate and meaningful for future supervised learning.

---

