# /figures — Clustering and Sensor Behavior Visualizations

This folder contains all visual outputs generated during Phase 1, Phase 2, and Phase 2.5 of the Hybrid Predictive Maintenance project. These plots provide visual evidence for how engines degrade over time, how clusters form in sensor space, and how preprocessing steps affect the data.

---

## Plot Categories and Descriptions

### 1. PCA Cluster Projections (`*_PCA_Clusters.png`)
- Shows clusters in 2D space using Principal Component Analysis.
- Helps visualize linear separability of degradation stages.

### 2. t-SNE Cluster Projections (`*_TSNE_Clusters.png`)
- Nonlinear projection using t-SNE to visualize local structure of cluster separation.
- Reveals fine-grained sensor behavior missed by PCA.

### 3. Cluster Distribution Bar Charts (`*_Cluster_Distribution.png`)
- Bar chart showing how many samples fall into each cluster/stage.
- Useful to check if clustering is balanced across stages.

### 4. Average Cycle per Stage (`*_Stage_TimeProfile.png`)
- Plots average engine cycle (time) for each KMeans stage.
- Validates that degradation stages follow logical time progression.

### 5. Elbow Method for KMeans (`*_ElbowCurve.png`)
- Plots WCSS (inertia) against different values of k (2 to 10).
- Helps identify the best number of clusters (we chose 5).

### 6. Hierarchical Dendrograms (`*_Dendrogram.png`)
- Dendrogram showing hierarchical relationships from Agglomerative clustering.
- Highlights similarity structure among engine samples.

### 7. Top Sensors per Stage (`*_TopSensors_PerStage.png`)
- Line plots of the top 3 high-variance sensors per dataset.
- Shows how those key sensors behave across each KMeans stage (0–4).

### 8. Correlation Heatmaps (Before vs After Normalization) (`*_Correlation_Before_After.png`)
- Side-by-side heatmaps of sensor correlation matrices:
  - Left: Before normalization (raw)
  - Right: After MinMax normalization
- Helps visualize the impact of preprocessing on sensor relationships.

### 9. Manual Cluster Verification Plots (`/manual_cluster_verification/FD00X/*.png`)
- Individual sensor behavior plots grouped by cluster stage after initial clustering.
- Used during Phase 2.5 to manually verify the correctness of unsupervised cluster stages.
- Helped identify stage overlaps and guided manual relabeling when needed.

---

## File Naming Format

Each file is named like:

| Format Component         | Meaning                                 |
|--------------------------|------------------------------------------|
| `FD001`                  | Dataset identifier (FD001 to FD004)      |
| `PCA_Clusters`           | PCA-based cluster visualization          |
| `TSNE_Clusters`          | t-SNE-based cluster visualization        |
| `Cluster_Distribution`   | Count of samples per cluster stage       |
| `Stage_TimeProfile`      | Avg. time value for each KMeans stage    |
| `ElbowCurve`             | WCSS vs. k curve (elbow method)          |
| `Dendrogram`             | Agglomerative tree structure             |
| `TopSensors_PerStage`    | Sensor trends for high-variance sensors  |
| `Correlation_Before_After` | Correlation matrix (raw vs. cleaned)  |
| `manual_cluster_verification/` | Per-sensor cluster behavior plots for manual inspection |

---

## Usage

These figures are used in:
- Clustering validation (Phase 2)
- Manual cluster stage verification and correction (Phase 2.5)
- Sensor diagnostics and selection (for Phase 3 modeling)
- Reporting and documentation
- Demonstrating preprocessing impact and model justification

