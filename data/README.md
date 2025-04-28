# /data — Dataset Repository

This folder contains all versions of the datasets used in Phases 1, 2, and 2.5 of the Hybrid Predictive Maintenance project. It includes the original NASA CMAPSS datasets, cleaned sensor data, clustered datasets, and manually verified corrected cluster datasets.

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

These CSV files contain the results of unsupervised clustering (KMeans and Agglomerative).  
Each file includes:
- Cluster IDs (`kmeans_cluster`, `agglo_cluster`)
- Degradation stage labels (`kmeans_stage`, `agglo_stage`)
- PCA and t-SNE projections (`pca_1`, `tsne_1`, etc.)

| File | Description |
|------|-------------|
| `clustered_train_FD001.csv` – `clustered_train_FD004.csv` | Sensor data with clustering and stage labels for each cycle |

Used in:  
- Visualizations (`/figures`)  
- Stage summaries (`/report/cluster_summaries/`)  
- As base for manual verification (Phase 2.5)

---

## 4. Corrected Clustered Datasets (Phase 2.5 Output)

After manually verifying degradation patterns in Phase 2.5, some datasets were relabeled to fix wrong or overlapping stage assignments.  
These corrected datasets reflect improved degradation ordering.

| File | Description |
|------|-------------|
| `corrected_clustered_train_FD001.csv` – `corrected_clustered_train_FD004.csv` | Clustered sensor data after manual relabeling or confirmation |

Used in:  
- Phase 3 supervised classification  
- Phase 4 RUL modeling  
- Ensures more reliable stage labels for training and testing

---

## Notes on Preprocessing Visuals

The impact of normalization is demonstrated visually in `/figures` as heatmaps showing **sensor correlation before and after scaling**.  
Similarly, manual cluster verification plots were generated in Phase 2.5 to justify the corrections made.

---

## Summary

This folder acts as the central data hub:
- Starts with raw CMAPSS files
- Produces cleaned and normalized datasets
- Adds unsupervised cluster labels
- Further improves stage labeling via manual verification

These versions ensure data quality and reliability across all future phases of the Hybrid Predictive Maintenance pipeline.

