# /figures — Clustering and Sensor Behavior Visualizations

This folder contains all visual plots generated during Phase 2 of the Hybrid Predictive Maintenance project. These visualizations are essential for analyzing how clustering results relate to sensor behavior and engine degradation over time. They also support interpretation, validation, and reporting in subsequent phases.

---

## Plot Categories and Descriptions

### 1. PCA Cluster Projections (`*_PCA_Clusters.png`)
- **Description:** Two-dimensional projection of clusters using Principal Component Analysis (PCA).
- **Purpose:** Allows visual inspection of how linearly separable the clusters are.
- **Generated using:** `sklearn.decomposition.PCA`

### 2. t-SNE Cluster Projections (`*_TSNE_Clusters.png`)
- **Description:** Non-linear 2D projection of clusters using t-SNE.
- **Purpose:** Highlights local structure and non-linear patterns in high-dimensional sensor data.
- **Generated using:** `sklearn.manifold.TSNE`

### 3. Cluster Distribution Bar Charts (`*_Cluster_Distribution.png`)
- **Description:** Shows the count of data points assigned to each degradation stage (Stage 0–4) for KMeans and Agglomerative clustering.
- **Purpose:** Helps assess class balance and validate clustering spread across stages.

### 4. Average Cycle per Stage (`*_Stage_TimeProfile.png`)
- **Description:** Plots the average engine cycle (time) for each cluster stage.
- **Purpose:** Confirms that cluster stages follow a progressive degradation timeline, supporting alignment with Remaining Useful Life (RUL) trends.

### 5. Elbow Curves (`*_ElbowCurve.png`)
- **Description:** Shows Within-Cluster Sum of Squares (WCSS) for different values of k (number of clusters).
- **Purpose:** Helps identify the optimal number of clusters using the "elbow method".

### 6. Hierarchical Dendrograms (`*_Dendrogram.png`)
- **Description:** Displays the hierarchical structure of clusters formed via Agglomerative Clustering.
- **Purpose:** Visualizes relationships and linkage between samples.
- **Generated using:** `scipy.cluster.hierarchy.dendrogram`

### 7. Top Sensors per Stage (`*_TopSensors_PerStage.png`)
- **Description:** Shows average values of the top 3 high-variance sensors across KMeans stages (Stage 0–4).
- **Purpose:** Highlights key sensors that evolve with degradation and supports feature selection for later phases.
- **Derived from:** `top_variance_sensors.json`

---

## File Naming Convention

Each file follows this structure:

| Component              | Description                                        |
|------------------------|----------------------------------------------------|
| `FD001`–`FD004`        | Corresponds to the dataset ID                      |
| `PCA_Clusters`         | Cluster projection using PCA                       |
| `TSNE_Clusters`        | Cluster projection using t-SNE                     |
| `Cluster_Distribution` | Count of data points per stage                     |
| `Stage_TimeProfile`    | Average cycle count per stage                      |
| `ElbowCurve`           | WCSS plotted for k = 2 to 10                       |
| `Dendrogram`           | Agglomerative cluster tree                         |
| `TopSensors_PerStage`  | Trends of most important sensors across clusters   |

---

## Usage

These plots are used throughout the project for:

- Evaluating clustering quality and degradation structure (Phase 2)
- Supporting stage-aware classification and regression (Phases 3–5)
- Enhancing interpretability in the final report
- Selecting relevant features based on sensor sensitivity
