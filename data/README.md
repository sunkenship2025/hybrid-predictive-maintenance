# /data — Dataset Repository

This folder contains all versions of the datasets used in Phases 1 and 2 of the Hybrid Predictive Maintenance project. It includes the original NASA CMAPSS datasets and the transformed versions created through preprocessing and clustering.

---

## 1. Raw Data Files (Original CMAPSS)

These are the original text files provided by NASA, containing engine sensor data under different operational conditions.

| File | Description |
|------|-------------|
| `train_FD001.txt` – `train_FD004.txt` | Training data with engine ID, cycle count, 3 operational settings, and 21 sensors |
| `test_FD001.txt` – `test_FD004.txt` | Test data for which the Remaining Useful Life (RUL) needs to be predicted |
| `RUL_FD001.txt` – `RUL_FD004.txt` | Ground-truth RUL values for each engine in the test set |

Used in: `01_data_cleaning.ipynb`

---

## 2. Cleaned Sensor Data (Phase 1 Output)

These files were generated after removing constant (zero-variance) sensors and applying MinMax normalization. Columns were renamed for clarity using `sensor_1`, `sensor_2`, etc.

| File | Description |
|------|-------------|
| `clean_train_FD001.csv` – `clean_train_FD004.csv` | Normalized versions of training datasets with constant sensors dropped |

Used in:  
- Clustering analysis (`02_clustering.ipynb`)  
- Heatmap correlation comparison (raw vs. cleaned)

---

## 3. Clustered Training Datasets (Phase 2 Output)

These CSV files contain the results of KMeans and Agglomerative clustering. Each file includes cluster labels, degradation stage labels, and 2D projections for PCA and t-SNE.

| File | Description |
|------|-------------|
| `clustered_train_FD001.csv` – `clustered_train_FD004.csv` | Sensor data with added columns: `kmeans_stage`, `agglo_stage`, `pca_1`, `tsne_1`, etc. |

Used in:  
- Visualization (`/figures`)  
- Stage-wise summaries (`/report/cluster_summaries/`)  
- Classification and RUL modeling in later phases

---

## Notes on Preprocessing Visuals

The impact of normalization is demonstrated visually in `/figures` as heatmaps showing **sensor correlation before and after scaling**. These are meant to explain how Phase 1 cleaning affects the structure and redundancy of features across datasets.

---

## Summary

This folder acts as the central data hub:
- Starts with raw CMAPSS files
- Produces cleaned, normalized datasets for analysis
- Adds cluster labels and projections for further modeling

It supports reproducibility and serves as the input foundation for all subsequent phases of the project.
