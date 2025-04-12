# /data — Dataset Repository

This folder contains all versions of the datasets used throughout Phases 1 and 2 of the Hybrid Predictive Maintenance project. It includes raw NASA CMAPSS datasets as well as processed and clustered versions created during preprocessing and unsupervised learning stages.

---

## 1. Raw Data Files (Original CMAPSS)

These files are sourced directly from the NASA CMAPSS dataset and represent multi-sensor time series data collected from aircraft engines under various operational conditions.

| File | Description |
|------|-------------|
| `train_FD001.txt` to `train_FD004.txt` | Raw training data containing engine unit number, cycle, 3 operational settings, and 21 sensor readings |
| `test_FD001.txt` to `test_FD004.txt` | Test data for which the Remaining Useful Life (RUL) needs to be predicted |
| `RUL_FD001.txt` to `RUL_FD004.txt` | Ground-truth RUL values corresponding to each engine unit in the test files |

These files serve as the foundation for all subsequent analysis and were directly processed in Phase 1 (`01_data_cleaning.ipynb`).

---

## 2. Cleaned Sensor Data (Phase 1 Output)

After preprocessing, the cleaned datasets contain:
- Dropped constant sensors (features with zero variance across all cycles)
- Normalized sensor values using MinMaxScaler
- Renamed columns for clarity (e.g., `sensor_1`, `sensor_2`, ..., `sensor_n`)

| File | Description |
|------|-------------|
| `clean_train_FD001.csv` |
| `clean_train_FD002.csv` |
| `clean_train_FD003.csv` |
| `clean_train_FD004.csv` |

These datasets serve as the input for Phase 2 clustering and subsequent modeling stages.

---

## 3. Clustered Training Datasets (Phase 2 Output)

These files contain the outputs of KMeans and Agglomerative clustering with assigned stage labels, as well as reduced feature projections for visualization.

Each file includes:
- `kmeans_cluster`, `agglo_cluster` — Numeric cluster labels (0 to 4)
- `kmeans_stage`, `agglo_stage` — Human-readable stage labels (Stage 0 to Stage 4)
- `pca_1`, `pca_2` — PCA components (2D projection)
- `tsne_1`, `tsne_2` — t-SNE components (nonlinear 2D projection)
- All original normalized sensor features

| File | Description |
|------|-------------|
| `clustered_train_FD001.csv` |
| `clustered_train_FD002.csv` |
| `clustered_train_FD003.csv` |
| `clustered_train_FD004.csv` |

These files are used in:
- Visualization of clustering results (PCA and t-SNE)
- Stage-wise performance evaluation and profiling
- Supervised learning (Phase 3 – Classification)
- RUL estimation within each degradation stage (Phase 4 and 5)
- Top 3 sensor trends per stage (`*_TopSensors_PerStage.png`)
- Average cycle time per stage (`*_Stage_TimeProfile.png`)
- Cluster summaries (`/report/cluster_summaries`)
