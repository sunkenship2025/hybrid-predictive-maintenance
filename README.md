# Hybrid Predictive Maintenance using Enhanced CMAPSS Dataset

This project builds a hybrid system to predict both engine degradation stages and Remaining Useful Life (RUL) using machine learning models on the NASA CMAPSS datasets.

---

## Project Structure

| Folder         | Description |
|----------------|-------------|
| `data/`        | Raw CMAPSS files, cleaned sensor data, and clustered datasets for FD001–FD004 |
| `figures/`     | Visualizations from Phase 1 and 2: PCA, t-SNE, cluster distributions, top sensor plots, heatmaps, dendrograms, elbow curves |
| `report/`      | Evaluation metrics, summaries (markdown and JSON), sensor variance rankings, cluster summaries |
| `notebooks/`   | Phase-wise Jupyter notebooks containing implementation and explanations |
| `scripts/`     | Python scripts exported from notebooks for reproducibility |
| `docs/`        | (Optional) Space to compile the final technical report using Typst or LaTeX |

---

## Project Phases

### Phase 1: Data Cleaning and Normalization
- Loaded CMAPSS raw datasets (FD001 to FD004)
- Assigned proper column names
- Dropped constant sensors (zero variance)
- Applied MinMaxScaler normalization to all sensor features
- Saved cleaned datasets for further use
- Generated **correlation heatmaps** to compare sensor relationships before and after normalization

### Phase 2: Clustering and Degradation Stage Labeling
- Performed unsupervised clustering using KMeans and Agglomerative Clustering (5 stages each)
- Validated clustering quality using **silhouette scores** and **elbow curves**
- Applied PCA and t-SNE to visualize sensor behavior in 2D
- Assigned cluster labels (`Stage 0` to `Stage 4`) for both methods
- Plotted:
  - Cluster projections (PCA/t-SNE)
  - Cluster distributions
  - Stage-wise average cycle (time)
  - High-variance sensor behavior per stage
  - Dendrograms (Agglomerative)
- Exported clustered datasets and summaries

### Phase 3 (Upcoming): Classification
- Will train supervised models to classify the degradation stage of each engine cycle using selected sensor features

---

## Key Files and Quick Access

### Data (`/data`)
- `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` — Original CMAPSS data
- `clean_train_FD00x.csv` — Normalized datasets after preprocessing
- `clustered_train_FD00x.csv` — Labeled with cluster IDs and stage names

### Notebooks (`/notebooks`)
- `01_data_cleaning.ipynb` — Phase 1: Preprocessing, normalization, heatmaps
- `02_clustering.ipynb` — Phase 2: Clustering, visualization, evaluation

### Visuals (`/figures`)
- PCA / t-SNE cluster projections
- Cluster distribution and stage-time trends
- Top sensor plots
- Elbow curves and dendrograms
- **New:** Correlation heatmaps (before vs. after normalization)

### Reports and Insights (`/report`)
- `silhouette_scores.json`, `silhouette_summary.md` — Clustering evaluation
- `elbow_scores.json`, `elbow_summary.md` — WCSS summary for KMeans
- `top_variance_sensors.json` — Top informative sensors
- `cluster_summaries/` — Mean sensor values per cluster stage

### Scripts (`/scripts`)
- `data_cleaning.py` — Reusable version of Phase 1 pipeline
- `clustering_analysis.py` — Phase 2 clustering and visual generation

---

## Contributors

See `/report/contributors.txt` for team member names.

---

## Final Report

A consolidated project report will be compiled at the end of Phase 5 using Typst or LaTeX.

---

## Deadline

Final submission due: **May 5**
