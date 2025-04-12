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
