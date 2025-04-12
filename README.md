# Hybrid Predictive Maintenance using Enhanced CMAPSS Dataset

This project implements a hybrid system that predicts both engine degradation stages and remaining useful life (RUL) using machine learning on the NASA CMAPSS datasets.

---

## Folder Structure

| Folder         | Description |
|----------------|-------------|
| `data/`        | Contains raw CMAPSS datasets, cleaned sensor data, and clustered outputs for FD001–FD004 |
| `figures/`     | All visualizations including PCA, t-SNE, cluster distributions, top sensor profiles, time profiles, dendrograms, and elbow curves |
| `report/`      | Markdown summaries, JSON evaluation metrics, top variance sensors, and stage-wise cluster statistics |
| `notebooks/`   | Phase-wise Jupyter Notebooks containing the main code and logic |
| `scripts/`     | Reproducible `.py` scripts exported from notebooks for automation and CLI execution |
| `docs/`        | (Optional) To be used for compiling the final report or paper using Typst or LaTeX |

---

## Project Phases

### Phase 1: Data Cleaning and Normalization
- Loaded all FD001–FD004 datasets from CMAPSS
- Dropped constant features based on zero variance
- Normalized all sensor readings using MinMaxScaler
- Saved cleaned datasets for reuse

### Phase 2: Clustering and Stage Labeling
- Applied KMeans and Agglomerative clustering (5 stages each)
- Used silhouette scores and WCSS (elbow method) to validate clustering quality
- Visualized clusters using PCA and t-SNE projections
- Created degradation stage labels and exported results
- Identified top high-variance sensors per dataset for later modeling
- Summarized average sensor values per cluster stage
- Generated stage-wise time profiles for RUL interpretation
- Exported evaluation metrics to Markdown (`silhouette_summary.md`, `elbow_summary.md`)
- Included dendrograms for hierarchical cluster interpretation

### Phase 3: Classification (Upcoming)
- Train classifiers to predict degradation stage using top sensor features

---

## Quick Access to Important Files

### Data and Processed Files (`/data`)
- `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` — original CMAPSS files
- `clean_train_FD00x.csv` — cleaned and normalized datasets
- `clustered_train_FD00x.csv` — labeled with KMeans and Agglomerative stages

### Notebooks (`/notebooks`)
- `01_data_cleaning.ipynb` — Phase 1 implementation
- `02_clustering.ipynb` — Phase 2 implementation with visualizations

### Visualizations (`/figures`)
- Cluster projections (`*_PCA_Clusters.png`, `*_TSNE_Clusters.png`)
- Cluster distribution plots (`*_Cluster_Distribution.png`)
- Sensor degradation behavior (`*_TopSensors_PerStage.png`)
- Time progression by stage (`*_Stage_TimeProfile.png`)
- Elbow and dendrogram plots for cluster validation

### Reports and Evaluation Files (`/report`)
- `silhouette_scores.json`, `silhouette_summary.md`
- `elbow_scores.json`, `elbow_summary.md`
- `top_variance_sensors.json`
- `cluster_summaries/` — average sensor values by stage per FD dataset

### Scripts (`/scripts`)
- `data_cleaning.py`
- `clustering_analysis.py`

---

## Contributors

Team member names are listed in `/report/contributors.txt`.

---

## Final Report

The final report will be compiled after Phase 5 using Typst.

---

## Timeline

Final submission deadline: May 5