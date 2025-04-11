# 📁 /data — All Dataset Versions

This folder contains **every dataset version** used or generated in Phases 1 and 2.

---

## 📂 Sections:

### 1️⃣ Raw Data Files (From NASA CMAPSS FD001–FD004)
These files contain real-world engine sensor readings with time-to-failure sequences.

| File | Content |
|------|---------|
| `train_FD00x.txt` | Raw training data: engine_id, cycle, 3 settings, 21 sensor readings |
| `test_FD00x.txt` | Test data for which RUL must be predicted |
| `RUL_FD00x.txt` | Remaining Useful Life values for the engines in the test set |

Used directly in `01_data_cleaning.ipynb`.

---

### 2️⃣ Cleaned Sensor Data (Output of Phase 1)
| File | Description |
|------|-------------|
| `clean_train_FD001.csv` – `clean_train_FD004.csv` | Normalized datasets after dropping constant sensors, with renamed columns for clarity (`sensor_1`, `sensor_2`, ...) |

These files serve as the **input to clustering and modeling stages**.

---

### 3️⃣ Clustered Datasets (Output of Phase 2)
| File | Description |
|------|-------------|
| `clustered_train_FD00x.csv` | Includes:
  - `kmeans_cluster`, `agglo_cluster`
  - Mapped labels: `kmeans_stage`, `agglo_stage` (Stage 0–4)
  - Visual features: `pca_1`, `pca_2`, `tsne_1`, `tsne_2` (optional)

These datasets are used in:
- 📊 Clustering visualizations
- 🧠 Classification model (Phase 3)
- 📈 RUL modeling per stage (Phase 4–5)
